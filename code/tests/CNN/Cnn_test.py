import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load Keras model
model = tf.keras.models.load_model("../../model_library/Models/Model-4/asl_cnn_model.h5")
class_names = [chr(i) for i in range(65, 91)]  # A-Z

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Webcam
cap = cv2.VideoCapture(0)
img_size = 64  # Should match training input size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box around hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = max(int(min(x_coords) * w) - 20, 0)
            y_min = max(int(min(y_coords) * h) - 20, 0)
            x_max = min(int(max(x_coords) * w) + 20, w)
            y_max = min(int(max(y_coords) * h) + 20, h)

            # Draw rectangle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Extract and preprocess ROI
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            roi_resized = cv2.resize(roi, (img_size, img_size))
            roi_normalized = roi_resized.astype("float32") / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)

            # Predict
            preds = model.predict(roi_input, verbose=0)
            pred_class = class_names[np.argmax(preds)]
            confidence = np.max(preds)
            print(preds)

            # Display prediction
            cv2.putText(frame, f'{pred_class} ({confidence:.2f})', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show output
    cv2.imshow("ASL Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()