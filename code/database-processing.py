import os
import cv2 as cv
import csv
import copy
import itertools
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import string

# Folder path
DATASET_DIR = "/home/amirthan/PycharmProjects/signlanguage_using_landmark/MyApp/ASL_Alphabet_Dataset/asl_alphabet_train"

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.7)

# Only process folders named 'A' to 'Z'
valid_letters = set(string.ascii_uppercase)
all_folders = sorted(os.listdir(DATASET_DIR))
letter_folders = [f for f in all_folders if f in valid_letters]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for idx in range(len(temp_landmark_list)):
        temp_landmark_list[idx][0] -= base_x
        temp_landmark_list[idx][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) or 1

    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

# Loop through each folder (A-Z)
for letter in tqdm(letter_folders, desc="Processing letters"):
    label_index = ord(letter) - ord('A') + 1  # 1 to 26
    letter_path = os.path.join(DATASET_DIR, letter)
    output_csv = os.path.join(letter_path, f"{letter}_labels.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"x{i}" for i in range(42)])  # 42 = 21 landmarks * 2

        images = [img for img in os.listdir(letter_path) if img.lower().endswith((".jpg", ".png", ".jpeg"))]

        for img_name in tqdm(images, desc=f"Processing {letter}", leave=False):
            img_path = os.path.join(letter_path, img_name)
            image = cv.imread(img_path)

            if image is None:
                continue

            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list = calc_landmark_list(image, hand_landmarks)
                processed = pre_process_landmark(landmark_list)
                writer.writerow([label_index] + processed)

hands.close()
print("\n✅ All letters processed successfully.")
