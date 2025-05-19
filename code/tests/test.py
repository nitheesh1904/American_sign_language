import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import copy
import argparse
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

# Utility functions

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


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

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    if max_value == 0:
        max_value = 1  # avoid division by zero

    temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
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
                if use_brect:
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                )

                # Calculate and print relative, normalized landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmarks = pre_process_landmark(landmark_list)
                hand_label = handedness.classification[0].label
                print(f"Relative Landmarks ({hand_label}):\n{pre_processed_landmarks}\n")

                # Display hand label
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                label_text = f"{label} ({int(score * 100)}%)"
                cv.putText(debug_image, label_text, (brect[0], brect[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow('Hand Tracking', debug_image)

        if cv.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
