import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
import argparse
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# ---------- Model and Label Encoder ----------
TFLITE_MODEL_PATH = "./model_library/Models/Model-3/asl_model.tflite"

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label encoder for A-Z
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([chr(i) for i in range(65, 91)])  # A-Z


# ---------- Utility Functions ----------

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.array([
        [min(int(lm.x * image_width), image_width - 1), min(int(lm.y * image_height), image_height - 1)]
        for lm in landmarks.landmark
    ])
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [
        [min(int(lm.x * image_width), image_width - 1), min(int(lm.y * image_height), image_height - 1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y

    flat_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, flat_list)) or 1
    return [n / max_value for n in flat_list]


def draw_prediction_text(image, text, position, color=(0, 0, 0)):
    cv.putText(image, text, position, cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)


def predict_asl_letter(landmark_vector):
    input_data = np.array([landmark_vector], dtype=np.float32)

    # Set tensor and invoke
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    confidence = output_data[0][predicted_class]
    if confidence < 0.75:
        predicted_label = "INVALID"
    return predicted_label, confidence


# ---------- Main Functionality ----------

def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    while True:
        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = image.copy()
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)

                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                processed_landmarks = pre_process_landmark(landmark_list)

                # Model prediction
                predicted_label, confidence = predict_asl_letter(processed_landmarks)

                # Display results
                if confidence > 0.75:
                    label_text = f"{predicted_label} ({int(confidence * 100)}%)"
                else:
                    label_text = f"{predicted_label}"
                draw_prediction_text(debug_image, label_text, (brect[0], brect[1] - 20))

        cv.imshow('ASL Real-Time Inference', debug_image)
        if cv.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    main()
