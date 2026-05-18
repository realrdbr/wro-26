"""
WRO 2026 - Future Engineers - Fischertechnik TXT 4.0
Autonomes Fahrzeug: Hindernisrennen mit KI-Erkennung
"""

import time
import sys
import cv2
import numpy as np

from fischertechnik.controller.Motor import Motor
from lib.controller import *

try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite


# =========================================================
# KONSTANTEN UND EINSTELLUNGEN
# =========================================================

# --- Lenkung (Servo) ---
CENTER       = 282    # Mittelstellung = geradeaus
MIN_STEERING = 182    # Maximaler Rechtseinschlag
MAX_STEERING = 382    # Maximaler Linkseinschlag

WALL_STEER_MAX_OFFSET = 100  

# --- Motor ---
NORMAL_SPEED   = 267   
CURVE_SPEED    = 200   
SLOW_SPEED     = 100   

# --- Kamera ---
CAMERA_WIDTH  = 320
CAMERA_HEIGHT = 240

# --- KI-Erkennung ---
DETECTION_THRESHOLD = 0.7   
CONFIRM_FRAMES_BLOCK = 2     
CONFIRM_FRAMES_LINE  = 1     

LABEL_GREEN  = "0 Gruen"    
LABEL_RED    = "1 Rot"       
LABEL_ROSA   = "2 Rosa"     
LABEL_LINE   = "3 Linien"   
LABEL_EMPTY  = "4 Leer"     

# --- Kurven-/Block-Timing (Sekunden) ---
BLOCK_TURN_MIN_DURATION = 0.4    
BLOCK_TURN_MAX_DURATION = 3.5    
CURVE_TURN_MIN_DURATION = 4.5    
CURVE_TURN_MAX_DURATION = 5      
STRAIGHTEN_DURATION     = 0.75   
# BEHOBEN: Cooldown verringert, damit die 3. Kurve nicht ignoriert wird!
TURN_COOLDOWN           = 1.50   

# --- Wandverfolgung ---
WALL_TARGET_DISTANCE   = 50.0   
WALL_TOLERANCE         = 7.0    
WALL_KP                = 3      
WALL_MIN               = 5      
WALL_MAX               = 200    
WALL_CORRECTION_DURATION = 0.12  
WALL_STRAIGHT_DURATION   = 0.10  

# --- Sensorglattung ---
SMOOTH_WINDOW = 7              
SENSOR_SAMPLE_INTERVAL = 0.015  

# --- Ultraschall: Kurvenrichtung ---
LINE_TURN_DECISION_GAP = 5.0   

# --- Rundenzaehlung ---
SECTIONS_PER_LAP    = 8     
TOTAL_LAPS          = 3     
TOTAL_SECTIONS      = SECTIONS_PER_LAP * TOTAL_LAPS  

CURVE_DISTANCE_THRESHOLD = 60  


# =========================================================
# HILFSFUNKTIONEN: SENSORGLATTUNG
# =========================================================

def median_filter(values):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        return float(sorted_vals[mid])


def clamp_sensor(value):
    if value <= 0:
        return float(WALL_MIN)
    if value > WALL_MAX:
        return float(WALL_MAX)
    return float(value)


def reject_outlier(new_val, buffer, max_jump=40):
    if len(buffer) < 3:
        return new_val
    med = median_filter(buffer)
    if abs(new_val - med) > max_jump:
        return med  
    return new_val


# =========================================================
# HILFSFUNKTIONEN: KI / KAMERA
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
# HILFSFUNKTIONEN: LENKUNG
# =========================================================

def clamp_steering(value):
    return int(max(MIN_STEERING, min(value, MAX_STEERING)))


def steer_towards(direction, intensity):
    intensity = max(0.0, min(intensity, 1.0))
    if direction == "left":
        return clamp_steering(CENTER + int((MAX_STEERING - CENTER) * intensity))
    else:
        return clamp_steering(CENTER - int((CENTER - MIN_STEERING) * intensity))


# =========================================================
# ZUSTANDSDEFINITIONEN
# =========================================================
STATE_STRAIGHT       = "GERADEAUS"         
STATE_BLOCK_AVOID    = "BLOCK_AUSWEICHEN"  
STATE_CURVE          = "KURVE_FAHREN"      
STATE_STRAIGHTEN     = "NACHLENKEN"        
STATE_WALL_CORRECT   = "WAND_KORREKTUR"    
STATE_WALL_STABILIZE = "WAND_STABILISIEREN"
STATE_STOPPED        = "GESTOPPT"          


# =========================================================
# HAUPTFUNKTION
# =========================================================

def main(argv):
    print("=== WRO 2026 Future Engineers - Start ===")

    dir_path   = "/opt/ft/workspaces"
    model_path = dir_path + "/model.tflite"
    label_path = dir_path + "/labels.txt"

    print("Lade Labels und Modell...")
    labels      = load_labels(label_path)
    interpreter = load_model(model_path)

    input_details = interpreter.get_input_details()
    input_shape   = input_details[0]["shape"]
    height        = input_shape[1]
    width         = input_shape[2]
    input_index   = input_details[0]["index"]

    print("Oeffne Kamera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Zustandskontrolle
    steering = CENTER
    TXT_M_S1_servomotor.set_position(int(steering))
    state = STATE_STRAIGHT

    left_buffer  = []
    right_buffer = []
    left_avg     = 50.0  
    right_avg    = 50.0

    last_sensor_time  = time.time()
    state_start_time  = time.time()   
    last_turn_time    = 0.0           
    state_timeout     = 0.0           

    drive_direction = None
    current_turn_dir = None   
    last_correction_dir = None

    confirm_label    = ""
    confirm_count    = 0
    confirmed_label  = ""
    confirmed_score  = 0.0

    sections_completed = 0   
    in_curve_zone      = False  
    loop_counter = 0

    # Initiale Sensormessung
    raw_left  = clamp_sensor(TXT_M_I1_ultrasonic_distance_meter.get_distance())
    raw_right = clamp_sensor(TXT_M_I2_ultrasonic_distance_meter.get_distance())
    left_buffer.append(raw_left)
    right_buffer.append(raw_right)

    while True:
        now = time.time()
        loop_counter += 1

        # 1. ULTRASCHALL LESEN
        if now - last_sensor_time >= SENSOR_SAMPLE_INTERVAL:
            last_sensor_time = now
            raw_left  = clamp_sensor(TXT_M_I1_ultrasonic_distance_meter.get_distance())
            raw_right = clamp_sensor(TXT_M_I2_ultrasonic_distance_meter.get_distance())

            raw_left  = reject_outlier(raw_left, left_buffer, max_jump=35)
            raw_right = reject_outlier(raw_right, right_buffer, max_jump=35)

            left_buffer.append(raw_left)
            right_buffer.append(raw_right)

            if len(left_buffer) > SMOOTH_WINDOW:
                left_buffer.pop(0)
            if len(right_buffer) > SMOOTH_WINDOW:
                right_buffer.pop(0)

            left_avg  = median_filter(left_buffer)
            right_avg = median_filter(right_buffer)

        # 2. KI-ERKENNUNG
        best_label = LABEL_EMPTY
        best_score = 0.0

        ret, frame = cap.read()
        if ret:
            image = cv2.resize(frame, (width, height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_dtype = input_details[0]["dtype"]
            scale, zero_point = input_details[0]["quantization"]

            if input_dtype == np.float32:
                image = image.astype(np.float32) / 255.0
            elif input_dtype == np.int8:
                image = (image / scale + zero_point).astype(np.int8)
            elif input_dtype == np.uint8:
                image = image.astype(np.uint8)

            top_result = process_image(interpreter, image, input_index, input_details)
            best_id, best_score = top_result[0]
            best_label = labels[best_id]

        if best_score >= DETECTION_THRESHOLD and best_label != LABEL_EMPTY:
            if best_label == confirm_label:
                confirm_count += 1
            else:
                confirm_label = best_label
                confirm_count = 1

            needed = CONFIRM_FRAMES_BLOCK
            if best_label == LABEL_LINE:
                needed = CONFIRM_FRAMES_LINE

            if confirm_count >= needed:
                confirmed_label = best_label
                confirmed_score = best_score
        else:
            if state == STATE_STRAIGHT:
                confirm_count   = 0
                confirm_label   = ""
                confirmed_label = ""
                confirmed_score = 0.0

        # 3. KURVENZONE ERKENNEN
        is_curve_zone = (left_avg > CURVE_DISTANCE_THRESHOLD or right_avg > CURVE_DISTANCE_THRESHOLD)

        if is_curve_zone and not in_curve_zone:
            sections_completed += 1
            in_curve_zone = True
            if loop_counter > 5:
                print(">> Kurvenzone betreten. Abschnitte: " + str(sections_completed))

        if not is_curve_zone and in_curve_zone:
            sections_completed += 1
            in_curve_zone = False
            print(">> Gerader Abschnitt. Abschnitte: " + str(sections_completed))

        if sections_completed >= TOTAL_SECTIONS and state != STATE_STOPPED:
            print("=== 3 RUNDEN GESCHAFFT -> STOPP ===")
            state = STATE_STOPPED

        # 4. FAHRTRICHTUNG ERKENNEN
        if drive_direction is None and is_curve_zone:
            if left_avg > CURVE_DISTANCE_THRESHOLD and right_avg < CURVE_DISTANCE_THRESHOLD:
                drive_direction = "ccw"
                print(">>> RICHTUNG ERKANNT: GEGEN UHRZEIGERSINN (ccw)")
            elif right_avg > CURVE_DISTANCE_THRESHOLD and left_avg < CURVE_DISTANCE_THRESHOLD:
                drive_direction = "cw"
                print(">>> RICHTUNG ERKANNT: IM UHRZEIGERSINN (cw)")

        # 5. ZUSTANDSMASCHINE
        if state == STATE_STOPPED:
            TXT_M_M1_encodermotor.set_speed(0, Motor.CCW)
            TXT_M_S1_servomotor.set_position(int(CENTER))
            time.sleep(0.1)
            continue

        elif state == STATE_STRAIGHT:
            cooldown_ok = (now - last_turn_time) > TURN_COOLDOWN

            if cooldown_ok and confirmed_label != "" and confirmed_score >= DETECTION_THRESHOLD:
                if confirmed_label == LABEL_RED:
                    print(">>> ROT ERKANNT -> RECHTS AUSWEICHEN")
                    state = STATE_BLOCK_AVOID
                    current_turn_dir = "right"
                    state_start_time = now
                    state_timeout    = now + BLOCK_TURN_MAX_DURATION
                    confirm_count    = 0
                    confirmed_label  = ""
                elif confirmed_label == LABEL_GREEN:
                    print(">>> GRUEN ERKANNT -> LINKS AUSWEICHEN")
                    state = STATE_BLOCK_AVOID
                    current_turn_dir = "left"
                    state_start_time = now
                    state_timeout    = now + BLOCK_TURN_MAX_DURATION
                    confirm_count    = 0
                    confirmed_label  = ""
                elif confirmed_label == LABEL_LINE:
                    if drive_direction == "cw":
                        turn_dir = "right"
                    elif drive_direction == "ccw":
                        turn_dir = "left"
                    else:
                        if left_avg - right_avg > LINE_TURN_DECISION_GAP:
                            turn_dir = "left"
                        elif right_avg - left_avg > LINE_TURN_DECISION_GAP:
                            turn_dir = "right"
                        else:
                            turn_dir = "left"

                    print(">>> LINIEN ERKANNT -> KURVE " + turn_dir.upper())
                    state = STATE_CURVE
                    current_turn_dir = turn_dir
                    state_start_time = now
                    state_timeout    = now + CURVE_TURN_MAX_DURATION
                    confirm_count    = 0
                    confirmed_label  = ""

            if state == STATE_STRAIGHT:
                error = left_avg - right_avg
                if abs(error) < WALL_TOLERANCE:
                    steering = CENTER
                    TXT_M_S1_servomotor.set_position(int(steering))
                    TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
                    TXT_M_M1_encodermotor.set_distance(int(100))
                else:
                    correction = int(error * WALL_KP)
                    correction = max(-WALL_STEER_MAX_OFFSET, min(correction, WALL_STEER_MAX_OFFSET))
                    steering = clamp_steering(CENTER + correction)

                    last_correction_dir = "left" if correction > 0 else "right"

                    TXT_M_S1_servomotor.set_position(int(steering))
                    TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
                    TXT_M_M1_encodermotor.set_distance(int(100))

                    state = STATE_WALL_CORRECT
                    state_start_time = now
                    state_timeout    = now + WALL_CORRECTION_DURATION

        elif state == STATE_WALL_CORRECT:
            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if now > state_timeout:
                state = STATE_WALL_STABILIZE
                state_start_time = now
                state_timeout    = now + WALL_STRAIGHT_DURATION

        elif state == STATE_WALL_STABILIZE:
            if last_correction_dir == "left":
                steering = clamp_steering(CENTER - 8)  
            elif last_correction_dir == "right":
                steering = clamp_steering(CENTER + 8)
            else:
                steering = CENTER

            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if now > state_timeout:
                steering = CENTER
                TXT_M_S1_servomotor.set_position(int(steering))
                state = STATE_STRAIGHT
                last_correction_dir = None

        elif state == STATE_BLOCK_AVOID:
            min_time_passed = (now - state_start_time) > BLOCK_TURN_MIN_DURATION
            block_visible = False
            if best_score >= DETECTION_THRESHOLD:
                if current_turn_dir == "right" and best_label == LABEL_RED:
                    block_visible = True
                elif current_turn_dir == "left" and best_label == LABEL_GREEN:
                    block_visible = True

            if block_visible:
                intensity = max(0.3, min((best_score - DETECTION_THRESHOLD) / (1.0 - DETECTION_THRESHOLD), 1.0))
                steering = steer_towards(current_turn_dir, intensity)
            else:
                steering = steer_towards(current_turn_dir, 0.85)

            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if (min_time_passed and not block_visible) or now > state_timeout:
                print(">> Block-Ausweichen beendet -> Nachlenken")
                state = STATE_STRAIGHTEN
                state_start_time = now
                state_timeout    = now + STRAIGHTEN_DURATION
                last_turn_time   = now
                last_correction_dir = current_turn_dir  

        elif state == STATE_CURVE:
            min_time_passed = (now - state_start_time) > CURVE_TURN_MIN_DURATION
            lines_visible = (best_score >= DETECTION_THRESHOLD and best_label == LABEL_LINE)

            steering = steer_towards(current_turn_dir, 0.9)
            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(CURVE_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if (min_time_passed and not lines_visible) or now > state_timeout:
                print(">> Kurve beendet -> Nachlenken")
                state = STATE_STRAIGHTEN
                state_start_time = now
                state_timeout    = now + STRAIGHTEN_DURATION
                last_turn_time   = now
                last_correction_dir = current_turn_dir

        elif state == STATE_STRAIGHTEN:
            opposite = "right" if last_correction_dir == "left" else "left"
            elapsed = now - state_start_time
            total   = STRAIGHTEN_DURATION

            if elapsed < total * 0.5:
                # BEHOBEN: Intensität von 0.01 auf 0.25 erhöht, um spürbar gegenzulenken!
                steering = steer_towards(opposite, 0.25)
            else:
                steering = CENTER

            TXT_M_S1_servomotor.set_position(int(steering))
            TXT_M_M1_encodermotor.set_speed(int(NORMAL_SPEED), Motor.CCW)
            TXT_M_M1_encodermotor.set_distance(int(100))

            if now > state_timeout:
                steering = CENTER
                TXT_M_S1_servomotor.set_position(int(steering))
                state = STATE_STRAIGHT
                
                # BEHOBEN: Die Puffer komplett leeren, um alte Kurven-Messdaten zu löschen
                left_buffer.clear()
                right_buffer.clear()
                
                confirmed_label = ""
                confirm_count   = 0

        if loop_counter % 10 == 0:
            runden = sections_completed // SECTIONS_PER_LAP
            rest   = sections_completed % SECTIONS_PER_LAP
            dir_str = str(drive_direction) if drive_direction else "?"
            print("L:" + str(round(left_avg, 1)) + " R:" + str(round(right_avg, 1)) + " KI:" + str(best_label) + " " + str(round(best_score, 2)) + " St:" + str(int(steering)) + " Z:" + str(state) + " Rd:" + str(runden) + "+" + str(rest) + " Dir:" + dir_str)

        time.sleep(0.03)

if __name__ == "__main__":
    main(sys.argv[1:])
