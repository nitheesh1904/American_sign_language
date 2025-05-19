import os
import cv2 as cv
import numpy as np
import copy
import itertools
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Optional: disable oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----------- Setup -----------
TFLITE_MODEL_PATH = "/model_library/Models/Model-3/asl_model.tflite"
OUTPUT_DIR = "live_prediction_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label encoder for A-Z
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([chr(i) for i in range(65, 91)])  # A-Z

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# ----------- Utility Functions -----------
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

def predict_asl_letter(landmark_vector):
    input_data = np.array([landmark_vector], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    confidence = output_data[0][predicted_class]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0] if confidence >= 0.75 else "INVALID"
    return predicted_label, confidence

# ----------- Webcam Loop -----------
cap = cv.VideoCapture(0)
frame_count = 0

print("📷 Press 'q' to quit recording...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    predicted = "INVALID"
    confidence = 0.0

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Bounding box
        bbox = calc_bounding_rect(frame, landmarks)
        cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        # Landmark vector
        landmark_list = calc_landmark_list(frame, landmarks)
        landmark_vector = pre_process_landmark(landmark_list)

        # Predict
        predicted, confidence = predict_asl_letter(landmark_vector)

    # Display text
    label_text = f"{predicted} ({confidence:.2f})" if predicted != "INVALID" else "INVALID"
    cv.putText(frame, label_text, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv.LINE_AA)

    # Save frame
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame{frame_count}_{predicted}_{confidence:.2f}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv.imwrite(filepath, frame)
    frame_count += 1

    # Show window
    cv.imshow("Live ASL Prediction", frame)

    # Exit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print(f"✅ Recording ended. Saved {frame_count} frames in '{OUTPUT_DIR}'.")
