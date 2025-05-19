import os
import cv2 as cv
import numpy as np
import shutil
import random
import copy
import itertools
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import mediapipe as mp

# ---------------------- Configuration ----------------------
TEST_DATASET = "./test_dataset"
SOURCE_DATASET = "./ASL_Alphabet_Dataset/asl_alphabet_train"  # Assumes each class folder a-z
OUTPUT_DATASET = "./output_dataset"
NUM_IMAGES_PER_CLASS = 10
TFLITE_MODEL_PATH = "/model_library/Models/Model-3/asl_model.tflite"

# ---------------------- Setup ----------------------
os.makedirs(TEST_DATASET, exist_ok=True)
os.makedirs(OUTPUT_DATASET, exist_ok=True)

for letter in [chr(i) for i in range(65, 91)]:  # a-z
    src_dir = os.path.join(SOURCE_DATASET, letter)
    dst_dir = os.path.join(TEST_DATASET, letter)
    os.makedirs(dst_dir, exist_ok=True)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sampled = random.sample(images, min(NUM_IMAGES_PER_CLASS, len(images)))
    for idx, img_name in enumerate(sampled):
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, f"{letter}{idx+1}.jpg")
        shutil.copy2(src_path, dst_path)

# ---------------------- Load Model ----------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array([chr(i) for i in range(65, 91)])  # A-Z

# ---------------------- MediaPipe Init ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ---------------------- Landmark Utils ----------------------
def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)] for lm in landmarks.landmark]

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y
    flat_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_val = max(map(abs, flat_list)) or 1
    return [n / max_val for n in flat_list]

def predict_asl_letter(landmark_vector):
    input_data = np.array([landmark_vector], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(output_data)
    confidence = output_data[0][pred_class]
    label = label_encoder.inverse_transform([pred_class])[0] if confidence >= 0.75 else "INVALID"
    return label, confidence

# ---------------------- Process Dataset ----------------------

for folder in os.listdir(TEST_DATASET):
    class_dir = os.path.join(TEST_DATASET, folder)
    if not os.path.isdir(class_dir): continue

    output_class_dir = os.path.join(OUTPUT_DATASET, folder)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(class_dir):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue

        img_path = os.path.join(class_dir, img_name)
        image = cv.imread(img_path)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            landmark_list = calc_landmark_list(image, result.multi_hand_landmarks[0])
            landmark_vector = pre_process_landmark(landmark_list)
            predicted_label, confidence = predict_asl_letter(landmark_vector)
        else:
            predicted_label, confidence = "NO_HAND", 0.0

        base_name = os.path.splitext(img_name)[0]
        out_name = f"{base_name}-{predicted_label}.jpg"
        output_path = os.path.join(output_class_dir, out_name)
        cv.imwrite(output_path, image)


print("✅ Done processing test dataset into output_dataset.")
