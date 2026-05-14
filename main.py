"""
WRO 2026 – Autonomes Fahrzeug (Fischertechnik TXT)

Dieses Skript steuert ein autonomes Modellfahrzeug auf Basis eines
Fischertechnik-TXT-Controllers. Es kombiniert:
  - Ultraschall-Wandverfolgung (links/rechts) für die Spurhaltung
  - KI-gestützte Farbblock-Erkennung (rot/grün) über TensorFlow Lite
    für gezielte Rechts-/Linksabbiegungen
  - Einen Servo für die Lenkung und einen Encoder-Motor für den Antrieb

Ablauf pro Schleifendurchlauf:
  1. Sensorwerte lesen und glätten
  2. Kamerabild auswerten (TFLite-Inferenz)
  3. Abbiege-Modus aktivieren, falls Farbbblock erkannt
  4. Aktiven Fahrmodus (Abbiegen / Geradeaus / Wandkorrektur) ausführen
  5. Debug-Ausgabe
"""

# Standardbibliotheken
import time   # Zeitfunktionen (time.time, time.sleep)
import sys    # Kommandozeilenargumente

# Bildverarbeitungs- und Numerik-Bibliotheken
import cv2        # OpenCV: Kameraauslese und Bildvorverarbeitung
import numpy as np  # NumPy: Array-Operationen für Bilddaten

# Fischertechnik-spezifische Bibliotheken
from fischertechnik.controller.Motor import Motor  # Motor-Konstanten (z. B. Motor.CCW)
from lib.controller import *                        # TXT-Controller-Objekte (Motoren, Sensoren, Servo)

# TensorFlow Lite: zuerst das vollständige Paket versuchen,
# bei ImportError auf die schlanke Runtime zurückfallen
try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite


# =========================================================
# EINSTELLUNGEN
# =========================================================

# --- Lenkung ---
CENTER       = 282   # Servo-Position für Geradeausfahrt
MIN_STEERING = 182   # Servo-Position für maximalen Rechtseinschlag
MAX_STEERING = 382   # Servo-Position für maximalen Linkseinschlag

# --- Motorgeschwindigkeit ---
NORMAL_SPEED = 200   # PWM-Wert für Geradeausfahrt
CURVE_SPEED  = 150   # PWM-Wert beim Abbiegen (langsamer für mehr Kontrolle)

# --- Kameraauflösung ---
CAMERA_WIDTH  = 320  # Aufnahmebreite in Pixeln
CAMERA_HEIGHT = 240  # Aufnahmehöhe in Pixeln

# --- KI-Erkennung ---
DETECTION_THRESHOLD = 0.70  # Mindest-Konfidenz (0–1), ab der ein Label als erkannt gilt

# --- Kurven-Timing (alle Angaben in Sekunden) ---
BLOCK_TURN_MIN_DURATION = 0.3   # Mindestlenkzeit; erst danach wird geprüft, ob der Block verschwunden ist
BLOCK_TURN_MAX_DURATION = 4.0   # Sicherheits-Timeout: Abbiegemodus wird spätestens nach dieser Zeit beendet
STRAIGHTEN_DURATION      = 0.3  # Dauer der kurzen Geradeausphase direkt nach dem Abbiegen
TURN_COOLDOWN            = 4.0  # Sperrzeit nach einer Kurve; verhindert sofortiges erneutes Abbiegen

# --- Wandverfolgung ---
WALL_TOLERANCE          = 3     # Toleranzband in cm; innerhalb dieser Differenz keine Korrektur
WALL_CORRECTION_DURATION = 0.10  # Dauer eines einzelnen Lenkkorrekturimpulses in Sekunden
WALL_STRAIGHT_DURATION  = 0.08  # Kurze Geradeausphase nach der Korrektur, um den Kurs zu stabilisieren
WALL_MIN                = 10    # Mindestwert für Ultraschall-Messung; ersetzt ungültige Werte ≤ 0
WALL_KP                 = 5.0   # Proportional-Verstärkung: cm Abweichung → Lenkeinheiten

# --- Ultraschall-Entscheidung für Kurvenrichtung ---
LINE_TURN_DECISION_GAP = 2.0  # Mindestdifferenz links/rechts in cm für sichere Richtungsentscheidung

# --- Label-Bezeichnungen (müssen exakt mit labels.txt übereinstimmen) ---
LABEL_RED   = "1 Rot"    # Label für roten Farbblock → rechts am Block vorbeifahren
LABEL_GREEN = "2 Grün"   # Label für grünen Farbblock → links am Block vorbeifahren
LABEL_LINE  = "3 Linie"  # Label für Ecklinie → Abbiegen per Ultraschall-Entscheidung
LABEL_EMPTY = "4 Leer"   # Label für leeres Bild → keine Sonderaktion, Spurhaltung bleibt aktiv

# --- Sensorglättung ---
SMOOTH_WINDOW = 5  # Anzahl der letzten Messwerte, über die der gleitende Mittelwert gebildet wird
SENSOR_SAMPLE_INTERVAL = 0.010  # Ultraschall-Abtastintervall in Sekunden (10 ms)


# =========================================================
# KAMERA / KI
# =========================================================

def load_labels(label_path):
    """Liest die Label-Datei zeilenweise ein und gibt eine Liste der Label-Strings zurück.

    Args:
        label_path (str): Pfad zur Textdatei mit einem Label pro Zeile.

    Returns:
        list[str]: Liste der Label-Namen ohne führende/nachfolgende Leerzeichen.
    """
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_model(model_path):
    """Lädt ein TensorFlow-Lite-Modell und reserviert Speicher für die Tensoren.

    Das Modell wird mit 4 parallelen Threads initialisiert, um die
    Inferenzzeit auf dem TXT-Controller zu minimieren.

    Args:
        model_path (str): Pfad zur .tflite-Modelldatei.

    Returns:
        tflite.Interpreter: Initialisierter Interpreter mit allozierten Tensoren.
    """
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


def block_score_factor(score):
    """Normiert den Erkennungswert auf [0, 1] relativ zum Schwellenwert.

    Ein Score genau am DETECTION_THRESHOLD ergibt 0.0, ein Score von 1.0
    ergibt 1.0. Damit kann der Lenkeinschlag proportional zur Nähe des
    Blocks skaliert werden (höhere Konfidenz ≈ Block ist näher).

    Args:
        score (float): Rohkonfidenz aus der TFLite-Inferenz (0.0 – 1.0).

    Returns:
        float: Normierter Faktor im Bereich [0.0, 1.0].
    """
    span = 1.0 - DETECTION_THRESHOLD  # Verfügbarer Wertebereich oberhalb der Schwelle
    if span <= 0:
        return 1.0  # Sonderfall: Schwelle = 1.0 → immer maximaler Einschlag
    return max(0.0, min((score - DETECTION_THRESHOLD) / span, 1.0))


def choose_line_turn_direction(left_avg, right_avg, last_direction):
    """Bestimmt die Kurvenrichtung aus Ultraschall-Abständen links/rechts.

    Args:
        left_avg (float): Geglätteter Abstand des linken Ultraschall-Sensors in cm.
        right_avg (float): Geglätteter Abstand des rechten Ultraschall-Sensors in cm.
        last_direction (str | None): Zuletzt gewählte Richtung ("left"/"right")
            als Fallback bei nahezu identischen Sensorwerten.

    Returns:
        str: "left" oder "right" als Zielrichtung für die Kurve.
    """
    delta = left_avg - right_avg
    if delta > LINE_TURN_DECISION_GAP:
        return "left"
    if delta < -LINE_TURN_DECISION_GAP:
        return "right"
    if last_direction in ("left", "right"):
        return last_direction
    return "left" if left_avg >= right_avg else "right"


def process_image(interpreter, image, input_index, input_details, k=3):
    """Führt eine TFLite-Inferenz auf dem übergebenen Bild durch.

    Das Bild wird in einen Batch-Tensor umgewandelt, an den Interpreter
    übergeben und ausgewertet. Quantisierte Modelle (int8/uint8) werden
    automatisch in float32-Konfidenzwerte umgerechnet.

    Args:
        interpreter: Initialisierter TFLite-Interpreter.
        image (np.ndarray): Vorverarbeitetes Eingabebild (H×W×3).
        input_index (int): Index des Eingabe-Tensors im Interpreter.
        input_details (list): Liste der Eingabe-Tensor-Metadaten.
        k (int): Anzahl der Top-Ergebnisse, die zurückgegeben werden sollen.

    Returns:
        list[tuple[int, float]]: Liste von (Label-ID, Konfidenz) für die k
        besten Treffer, absteigend nach Konfidenz sortiert.
    """
    # Bild um eine Batch-Dimension erweitern: (H, W, C) → (1, H, W, C)
    input_data = np.expand_dims(image, axis=0)

    # Eingabe-Tensor befüllen und Inferenz starten
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Ausgabe-Tensor auslesen
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Quantisierung rückgängig machen, falls das Modell int8/uint8 ausgibt
    output_scale, output_zero_point = output_details[0]["quantization"]
    if output_details[0]["dtype"] in [np.int8, np.uint8]:
        output_data = (
            output_data.astype(np.float32) - output_zero_point
        ) * output_scale

    # Batch-Dimension entfernen und die k größten Konfidenzwerte ermitteln
    output_data = np.squeeze(output_data)
    top_k = output_data.argsort()[-k:][::-1]  # Indizes absteigend sortiert
    return [(_id, float(output_data[_id])) for _id in top_k]


# =========================================================
# MAIN
# =========================================================

def main(argv):
    """Hauptfunktion: Initialisierung und Steuerungsschleife des Fahrzeugs.

    Lädt das KI-Modell und die Labels, öffnet die Kamera, setzt alle
    Zustandsvariablen und startet die Echtzeit-Steuerungsschleife.

    Args:
        argv (list): Kommandozeilenargumente (derzeit nicht verwendet).
    """
    # --- Modell- und Label-Pfade ---
    dir_path = "/opt/ft/workspaces"        # Basisverzeichnis auf dem TXT-Controller
    model_path = dir_path + "/model.tflite"  # TFLite-Modell für Farberkennung
    label_path = dir_path + "/labels.txt"    # Zugehörige Label-Datei

    # Modell und Labels laden
    labels      = load_labels(label_path)
    interpreter = load_model(model_path)

    # Eingabe-Tensor-Metadaten auslesen (Form, Datentyp, Index)
    input_details = interpreter.get_input_details()
    input_shape   = input_details[0]["shape"]  # [Batch, Höhe, Breite, Kanäle]
    height        = input_shape[1]             # Erwartete Bildhöhe des Modells
    width         = input_shape[2]             # Erwartete Bildbreite des Modells
    input_index   = input_details[0]["index"]  # Tensor-Index für set_tensor()

    # --- Kamera initialisieren ---
    cap = cv2.VideoCapture(0)                              # Erste angeschlossene Kamera öffnen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)       # Aufnahmebreite setzen
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)      # Aufnahmehöhe setzen
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                    # Puffer auf 1 Frame → immer aktuellstes Bild

    # =====================================================
    # STARTWERTE
    # =====================================================

    # Lenkung auf Mittelstellung (Geradeaus) initialisieren
    steering = CENTER
    TXT_M_S1_servomotor.set_position(int(steering))

    # Puffer für gleitende Mittelwerte der Ultraschall-Sensoren
    left_values  = []  # Letzte SMOOTH_WINDOW Messwerte des linken Sensors
    right_values = []  # Letzte SMOOTH_WINDOW Messwerte des rechten Sensors

    # Abbiegemodi: werden aktiviert, wenn ein Farbblock erkannt wird
    # (Rot → rechts am Block vorbei, Grün → links am Block vorbei)
    turn_left_mode  = False  # Aktiv während Linksabbiegung
    turn_right_mode = False  # Aktiv während Rechtsabbiegung
    turn_end_time   = 0.0    # Zeitpunkt, ab dem Sichtbarkeit geprüft wird
    turn_max_time   = 0.0    # Sicherheits-Timeout für den Abbiegevorgang

    # Geradeausphase direkt nach einer Kurve (verhindert sofortige Wandkorrektur)
    straighten_mode    = False  # Aktiv während der Post-Kurven-Geradeausphase
    straighten_end_time = 0.0   # Zeitpunkt, an dem diese Phase endet

    # Wandkorrektur-Modi: werden abwechselnd aktiviert bei Abweichung vom Mittelstreifen
    wall_correction_mode = False  # Aktiv während des Lenkkorrekturimpulses
    wall_correction_end  = 0.0    # Zeitpunkt, an dem die Korrektur endet

    wall_straight_mode = False  # Aktiv während der Stabilisierungsphase nach Korrektur
    wall_straight_end  = 0.0    # Zeitpunkt, an dem die Stabilisierung endet

    # Zeitstempel der zuletzt abgeschlossenen Kurve (für TURN_COOLDOWN)
    last_turn_time = 0.0

    # Zuletzt verwendete Linien-Kurvenrichtung (Fallback bei nahezu gleichen Sensorwerten)
    last_line_turn_direction = None

    # Initiale Ultraschall-Messung für gültige Startwerte
    left = TXT_M_I1_ultrasonic_distance_meter.get_distance()
    right = TXT_M_I2_ultrasonic_distance_meter.get_distance()
    if left <= 0:
        left = WALL_MIN
    if right <= 0:
        right = WALL_MIN
    left_values.append(left)
    right_values.append(right)
    left_avg = float(left)
    right_avg = float(right)
    # Zeitbasierte Ultraschall-Abtastung: letzte geglättete Werte zwischen Abtastungen wiederverwenden
    last_sensor_sample_time = time.time()

    # =====================================================
    # HAUPTSCHLEIFE
    # =====================================================
    while True:
        now = time.time()  # Aktuellen Zeitstempel für alle zeitbasierten Vergleiche

        # =================================================
        # SENSORWERTE LESEN
        # =================================================

        # Ultraschall nur alle SENSOR_SAMPLE_INTERVAL neu lesen; dazwischen letzte geglättete Werte nutzen
        if now - last_sensor_sample_time >= SENSOR_SAMPLE_INTERVAL:
            last_sensor_sample_time = now

            # Rohmesswerte von den Ultraschall-Abstandssensoren auslesen
            left  = TXT_M_I1_ultrasonic_distance_meter.get_distance()  # Linker Sensor (cm)
            right = TXT_M_I2_ultrasonic_distance_meter.get_distance()  # Rechter Sensor (cm)

            # Ungültige Messwerte (Sensor nicht im Messbereich) durch Mindestwert ersetzen
            if left  <= 0:
                left  = WALL_MIN
            if right <= 0:
                right = WALL_MIN

            # Neue Werte in die Glättungspuffer aufnehmen
            left_values.append(left)
            right_values.append(right)

            # Puffer auf SMOOTH_WINDOW begrenzen (älteste Werte entfernen)
            if len(left_values) > SMOOTH_WINDOW:
                left_values.pop(0)
                right_values.pop(0)

            # Gleitenden Mittelwert berechnen
            left_avg  = sum(left_values) / len(left_values)
            right_avg = sum(right_values) / len(right_values)

        # =================================================
        # KAMERA / KI
        # =================================================

        # Standardwerte für den Fall, dass kein Frame gelesen werden kann
        best_label = ""
        best_score = 0.0

        ret, frame = cap.read()  # Kamerabild lesen; ret=False bei Fehler
        if ret:
            # Bild auf die vom Modell erwartete Größe skalieren
            image = cv2.resize(frame, (width, height))
            # OpenCV liefert BGR, TFLite-Modell erwartet RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Datentyp und Quantisierungsparameter des Eingabe-Tensors auslesen
            input_dtype         = input_details[0]["dtype"]
            scale, zero_point   = input_details[0]["quantization"]

            # Bild in den Datentyp des Modells konvertieren
            if input_dtype == np.float32:
                # Float-Modell: Pixelwerte auf [0.0, 1.0] normieren
                image = image.astype(np.float32) / 255.0
            elif input_dtype == np.int8:
                # Quantisiertes int8-Modell: Pixelwerte mit Skalierungsformel umrechnen
                image = (image / scale + zero_point).astype(np.int8)

            # Inferenz durchführen und bestes Ergebnis extrahieren
            top_result        = process_image(interpreter, image, input_index, input_details)
            best_id, best_score = top_result[0]   # Label-ID und Konfidenz des Top-Treffers
            best_label        = labels[best_id]   # Lesbarer Label-Name

        # =================================================
        # ROT / GRÜN ERKENNUNG → ABBIEGEN EINLEITEN
        # =================================================

        # Cooldown-Prüfung: genug Zeit seit letzter Kurve vergangen?
        cooldown_ok  = (now - last_turn_time) > TURN_COOLDOWN
        # Modus-Prüfung: kein aktiver Abbiegevorgang oder Geradeausphase?
        not_in_turn  = not turn_left_mode and not turn_right_mode and not straighten_mode

        # Nur wenn alle Bedingungen erfüllt und ein Block mit ausreichender Konfidenz erkannt:
        if cooldown_ok and not_in_turn and best_score >= DETECTION_THRESHOLD:
            if best_label == LABEL_RED:
                # Roter Block erkannt → rechts am Block vorbeifahren
                print(">>> ROT ERKANNT → RECHTS ABBIEGEN")
                turn_right_mode = True
                turn_end_time  = now + BLOCK_TURN_MIN_DURATION  # Frühester Zeitpunkt für Beendigung
                turn_max_time  = now + BLOCK_TURN_MAX_DURATION  # Sicherheits-Timeout
                last_turn_time = now                             # Cooldown-Uhr starten

            elif best_label == LABEL_GREEN:
                # Grüner Block erkannt → links am Block vorbeifahren
                print(">>> GRÜN ERKANNT → LINKS ABBIEGEN")
                turn_left_mode = True
                turn_end_time   = now + BLOCK_TURN_MIN_DURATION
                turn_max_time   = now + BLOCK_TURN_MAX_DURATION
                last_turn_time  = now

            elif best_label == LABEL_LINE:
                # Ecklinie erkannt → Richtung aus Ultraschall links/rechts bestimmen
                line_turn_direction = choose_line_turn_direction(
                    left_avg, right_avg, last_line_turn_direction
                )
                if line_turn_direction == "right":
                    print(">>> LINIE ERKANNT (US) → RECHTS ABBIEGEN")
                    turn_right_mode = True
                else:
                    print(">>> LINIE ERKANNT (US) → LINKS ABBIEGEN")
                    turn_left_mode  = True
                last_line_turn_direction = line_turn_direction
                turn_end_time  = now + BLOCK_TURN_MIN_DURATION
                turn_max_time  = now + BLOCK_TURN_MAX_DURATION
                last_turn_time = now

            elif best_label == LABEL_EMPTY:
                # Leeres Bild erkannt → keine Sonderaktion; Spurhaltung (Wandverfolgung) bleibt aktiv
                print(">>> LEER ERKANNT → SPURHALTUNG AKTIV")

        # =================================================
        # LINKS ABBIEGEN (Rot-Klotz)
        # =================================================

        if turn_left_mode:
            print("turn_left_mode")
            # Prüfen, ob das erkannte Objekt (grüner Block oder Ecklinie) noch sichtbar ist
            target_still_visible = (
                best_score >= DETECTION_THRESHOLD
                and best_label in (LABEL_GREEN, LABEL_LINE)
            )

            if target_still_visible:
                # Proportional: höhere Konfidenz (= näher) → stärkerer Lenkeinschlag nach links
                steering = int(CENTER + (MAX_STEERING - CENTER) * block_score_factor(best_score))
            else:
                # Objekt temporär verdeckt (z. B. durch Karosserie) – voller Einschlag beibehalten
                steering = MAX_STEERING

            # Lenkwinkel auf gültigen Bereich begrenzen und Servo setzen
            steering = max(MIN_STEERING, min(steering, MAX_STEERING))
            TXT_M_S1_servomotor.set_position(int(steering))
            # Motor mit Kurvengeschwindigkeit entgegen dem Uhrzeigersinn antreiben
            TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            # Abbiegung beenden, wenn Objekt verschwunden (nach Mindestzeit) oder Sicherheits-Timeout
            if (now > turn_end_time and not target_still_visible) or now > turn_max_time:
                turn_left_mode     = False
                straighten_mode    = True                         # Geradeausphase starten
                straighten_end_time = now + STRAIGHTEN_DURATION

        # =================================================
        # RECHTS ABBIEGEN (Grün-Klotz)
        # =================================================

        elif turn_right_mode:
            print("turn_right_mode")
            # Prüfen, ob das erkannte Objekt (roter Block oder Ecklinie) noch sichtbar ist
            target_still_visible = (
                best_score >= DETECTION_THRESHOLD
                and best_label in (LABEL_RED, LABEL_LINE)
            )

            if target_still_visible:
                # Proportional: höhere Konfidenz (= näher) → stärkerer Lenkeinschlag nach rechts
                steering = int(CENTER - (CENTER - MIN_STEERING) * block_score_factor(best_score))
            else:
                # Objekt temporär verdeckt – voller Einschlag beibehalten
                steering = MIN_STEERING

            # Lenkwinkel auf gültigen Bereich begrenzen und Servo setzen
            steering = max(MIN_STEERING, min(steering, MAX_STEERING))
            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            # Abbiegung beenden, wenn Objekt verschwunden (nach Mindestzeit) oder Sicherheits-Timeout
            if (now > turn_end_time and not target_still_visible) or now > turn_max_time:
                turn_right_mode    = False
                straighten_mode    = True                         # Geradeausphase starten
                straighten_end_time = now + STRAIGHTEN_DURATION

        # =================================================
        # KURZE GERADEAUSPHASE NACH KURVE
        # =================================================

        elif straighten_mode:
            print("straighten_mode")
            # Servo auf Mittelstellung setzen (Geradeaus)
            steering = CENTER

            TXT_M_S1_servomotor.set_position(int(steering))
            # Motor mit Normalgeschwindigkeit fahren
            TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            # Nach Ablauf der Geradeausphase alle Wandkorrektur-Zustände zurücksetzen
            if now > straighten_end_time:
                straighten_mode      = False
                wall_correction_mode = False  # Laufende Korrektur verwerfen
                wall_straight_mode   = False  # Laufende Stabilisierung verwerfen

        # =================================================
        # WANDVERFOLGUNG (Normalmodus)
        # =================================================

        else:
            # Regelabweichung: positive Werte → links weiter weg → nach links lenken
            error = left_avg - right_avg

            if wall_correction_mode:
                # Korrekturimpuls ist aktiv: eingestellten Lenkwinkel halten
                print("wall_correction_mode")

                TXT_M_S1_servomotor.set_position(int(steering))
                TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
                TXT_M_M1_encodermotor.set_distance(int(100))

                # Korrektur beendet → kurze Stabilisierungsphase starten
                if now > wall_correction_end:
                    wall_correction_mode = False
                    wall_straight_mode   = True
                    wall_straight_end    = now + WALL_STRAIGHT_DURATION

            elif wall_straight_mode:
                # Stabilisierungsphase: geradeaus fahren, damit das Fahrzeug ausschwingt
                print("wall_straight_mode")
                steering = CENTER

                TXT_M_S1_servomotor.set_position(int(steering))
                TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
                TXT_M_M1_encodermotor.set_distance(int(100))

                # Stabilisierung beendet → zurück zur normalen Fehlerberechnung
                if now > wall_straight_end:
                    wall_straight_mode = False

            else:
                if abs(error) < WALL_TOLERANCE:
                    # Abweichung innerhalb der Toleranz → keine Korrektur nötig
                    steering = CENTER
                    print("steering=center")

                elif error > 0:
                    # Linke Wand weiter entfernt → Fahrzeug zu nah an der rechten Wand
                    # → Lenkkorrektur nach links (steering > CENTER)
                    correction = min(int(abs(error) * WALL_KP), MAX_STEERING - CENTER)
                    steering = CENTER + correction
                    print(f"steering=links +{correction}")
                    wall_correction_mode = True
                    wall_correction_end  = now + WALL_CORRECTION_DURATION

                else:
                    # Rechte Wand weiter entfernt → Fahrzeug zu nah an der linken Wand
                    # → Lenkkorrektur nach rechts (steering < CENTER)
                    correction = min(int(abs(error) * WALL_KP), CENTER - MIN_STEERING)
                    steering = CENTER - correction
                    print(f"steering=rechts -{correction}")
                    wall_correction_mode = True
                    wall_correction_end  = now + WALL_CORRECTION_DURATION

                # Lenkwinkel auf gültigen Bereich begrenzen, Servo und Motor setzen
                steering = max(MIN_STEERING, min(steering, MAX_STEERING))
                TXT_M_S1_servomotor.set_position(int(steering))
                TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
                TXT_M_M1_encodermotor.set_distance(int(100))

        # =================================================
        # DEBUG
        # =================================================

        # Kompakte Statuszeile: Sensorwerte, KI-Ergebnis und aktueller Lenkwinkel
        print(
            "L:", round(left_avg, 1),
            "R:", round(right_avg, 1),
            "Label:", best_label,
            "Score:", round(best_score, 2),
            "Steering:", int(steering),
        )

        # Kurze Pause, um CPU-Last zu begrenzen (~33 Zyklen/Sekunde)
        time.sleep(0.03)


# Einstiegspunkt: main() nur aufrufen, wenn dieses Skript direkt gestartet wird
if __name__ == "__main__":
    main(sys.argv[1:])
