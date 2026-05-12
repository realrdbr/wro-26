import time
import cv2
import numpy as np
import sys

from fischertechnik.controller.Motor import Motor
from lib.controller import *

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite


# =========================================================
# EINSTELLUNGEN
# =========================================================

CENTER       = 282   # Lenkung gerade aus
MIN_STEERING = 182   # Vollausschlag rechts
MAX_STEERING = 382   # Vollausschlag links

NORMAL_SPEED = 200   # Geschwindigkeit Geradeaus
CURVE_SPEED  = 150   # Geschwindigkeit beim Abbiegen

CAMERA_WIDTH  = 320
CAMERA_HEIGHT = 240

# KI-Erkennung
DETECTION_THRESHOLD = 0.70  # Mindest-Konfidenz für Farberkennung

# Kurven-Timing (Sekunden)
TURN_DURATION      = 2.5   # Dauer des Lenkeinschlags beim Abbiegen
STRAIGHTEN_DURATION = 0.3  # Kurze Geradeausphase nach dem Abbiegen
TURN_COOLDOWN      = 4.0   # Sperrzeit nach einer Kurve (kein Re-Trigger)

# Wandverfolgung
WALL_TOLERANCE          = 3     # cm Toleranz links/rechts bevor korrigiert wird
WALL_CORRECTION_DURATION = 0.10  # Sekunden mit Lenkkorrektur
WALL_STRAIGHT_DURATION  = 0.08  # Sekunden Geradeaus zum Stabilisieren
WALL_MIN                = 10    # Mindestwert für Ultraschall-Sensor (ersetzt <=0)

# Glättungsfenster für Sensorwerte
SMOOTH_WINDOW = 5

# Label-Namen (müssen mit labels.txt übereinstimmen)
LABEL_RED   = "1 Rot"
LABEL_GREEN = "2 Grün"


# =========================================================
# KAMERA / KI
# =========================================================

def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index, input_details, k=3):
    input_data = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    output_scale, output_zero_point = output_details[0]["quantization"]
    if output_details[0]["dtype"] in [np.int8, np.uint8]:
        output_data = (
            output_data.astype(np.float32) - output_zero_point
        ) * output_scale

    output_data = np.squeeze(output_data)
    top_k = output_data.argsort()[-k:][::-1]
    return [(_id, float(output_data[_id])) for _id in top_k]


# =========================================================
# MAIN
# =========================================================

def main(argv):
    dir_path = "/opt/ft/workspaces"
    model_path = dir_path + "/model.tflite"
    label_path = dir_path + "/labels.txt"

    labels      = load_labels(label_path)
    interpreter = load_model(model_path)

    input_details = interpreter.get_input_details()
    input_shape   = input_details[0]["shape"]
    height        = input_shape[1]
    width         = input_shape[2]
    input_index   = input_details[0]["index"]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # =====================================================
    # STARTWERTE
    # =====================================================

    steering = CENTER
    TXT_M_S1_servomotor.set_position(int(steering))

    left_values  = []
    right_values = []

    # Abbiegemodi (Rot → links, Grün → rechts)
    turn_left_mode  = False
    turn_right_mode = False
    turn_end_time   = 0.0

    # Kurze Geradeausphase nach dem Abbiegen
    straighten_mode    = False
    straighten_end_time = 0.0

    # Wandkorrektur-Modus
    wall_correction_mode = False
    wall_correction_end  = 0.0

    wall_straight_mode = False
    wall_straight_end  = 0.0

    # Zeitstempel der letzten Kurve (für Cooldown)
    last_turn_time = 0.0

    while True:
        now = time.time()

        # =================================================
        # SENSORWERTE LESEN
        # =================================================

        left  = TXT_M_I1_ultrasonic_distance_meter.get_distance()
        right = TXT_M_I2_ultrasonic_distance_meter.get_distance()

        if left  <= 0:
            left  = WALL_MIN
        if right <= 0:
            right = WALL_MIN

        left_values.append(left)
        right_values.append(right)

        if len(left_values) > SMOOTH_WINDOW:
            left_values.pop(0)
            right_values.pop(0)

        left_avg  = sum(left_values) / len(left_values)
        right_avg = sum(right_values) / len(right_values)

        # =================================================
        # KAMERA / KI
        # =================================================

        best_label = ""
        best_score = 0.0

        ret, frame = cap.read()
        if ret:
            image = cv2.resize(frame, (width, height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_dtype         = input_details[0]["dtype"]
            scale, zero_point   = input_details[0]["quantization"]

            if input_dtype == np.float32:
                image = image.astype(np.float32) / 255.0
            elif input_dtype == np.int8:
                image = (image / scale + zero_point).astype(np.int8)

            top_result        = process_image(interpreter, image, input_index, input_details)
            best_id, best_score = top_result[0]
            best_label        = labels[best_id]

        # =================================================
        # ROT / GRÜN ERKENNUNG → ABBIEGEN EINLEITEN
        # =================================================

        cooldown_ok  = (now - last_turn_time) > TURN_COOLDOWN
        not_in_turn  = not turn_left_mode and not turn_right_mode and not straighten_mode

        if cooldown_ok and not_in_turn and best_score >= DETECTION_THRESHOLD:
            if best_label == LABEL_RED:
                print(">>> ROT ERKANNT → LINKS ABBIEGEN")
                turn_left_mode = True
                turn_end_time  = now + TURN_DURATION
                last_turn_time = now

            elif best_label == LABEL_GREEN:
                print(">>> GRÜN ERKANNT → RECHTS ABBIEGEN")
                turn_right_mode = True
                turn_end_time   = now + TURN_DURATION
                last_turn_time  = now

        # =================================================
        # LINKS ABBIEGEN (Rot-Klotz)
        # =================================================

        if turn_left_mode:
            print("turn_left_mode")
            steering = MAX_STEERING  # 382 = voller Linkslenkeinschlag

            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if now > turn_end_time:
                turn_left_mode     = False
                straighten_mode    = True
                straighten_end_time = now + STRAIGHTEN_DURATION

        # =================================================
        # RECHTS ABBIEGEN (Grün-Klotz)
        # =================================================

        elif turn_right_mode:
            print("turn_right_mode")
            steering = MIN_STEERING  # 182 = voller Rechtslenkeinschlag

            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if now > turn_end_time:
                turn_right_mode    = False
                straighten_mode    = True
                straighten_end_time = now + STRAIGHTEN_DURATION

        # =================================================
        # KURZE GERADEAUSPHASE NACH KURVE
        # =================================================

        elif straighten_mode:
            print("straighten_mode")
            steering = CENTER

            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if now > straighten_end_time:
                straighten_mode      = False
                # Wandkorrektur-Zustände zurücksetzen
                wall_correction_mode = False
                wall_straight_mode   = False

        # =================================================
        # WANDVERFOLGUNG (Normalmodus)
        # =================================================

        else:
            error = left_avg - right_avg  # > 0: links weiter weg → nach links lenken

            if wall_correction_mode:
                print("wall_correction_mode")

                TXT_M_S1_servomotor.set_position(int(steering))
                TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
                TXT_M_M1_encodermotor.set_distance(int(100))

                if now > wall_correction_end:
                    wall_correction_mode = False
                    wall_straight_mode   = True
                    wall_straight_end    = now + WALL_STRAIGHT_DURATION

            elif wall_straight_mode:
                print("wall_straight_mode")
                steering = CENTER

                TXT_M_S1_servomotor.set_position(int(steering))
                TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
                TXT_M_M1_encodermotor.set_distance(int(100))

                if now > wall_straight_end:
                    wall_straight_mode = False

            else:
                if abs(error) < WALL_TOLERANCE:
                    # Innerhalb der Toleranz → geradeaus
                    steering = CENTER
                    print("steering=center")

                elif error > 0:
                    # Linke Wand weiter entfernt → Roboter zu nah an rechter Wand → nach links korrigieren
                    steering = CENTER + 40
                    print("steering=links +40")
                    wall_correction_mode = True
                    wall_correction_end  = now + WALL_CORRECTION_DURATION

                else:
                    # Rechte Wand weiter entfernt → Roboter zu nah an linker Wand → nach rechts korrigieren
                    steering = CENTER - 40
                    print("steering=rechts -40")
                    wall_correction_mode = True
                    wall_correction_end  = now + WALL_CORRECTION_DURATION

                steering = max(MIN_STEERING, min(steering, MAX_STEERING))
                TXT_M_S1_servomotor.set_position(int(steering))
                TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
                TXT_M_M1_encodermotor.set_distance(int(100))

        # =================================================
        # DEBUG
        # =================================================

        print(
            "L:", round(left_avg, 1),
            "R:", round(right_avg, 1),
            "Label:", best_label,
            "Score:", round(best_score, 2),
            "Steering:", int(steering),
        )

        time.sleep(0.03)


if __name__ == "__main__":
    main(sys.argv[1:])
