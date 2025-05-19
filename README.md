# Real-Time American Sign Language Recognition Using ANN and MediaPipe

This project implements a real-time system to recognize American Sign Language (ASL) gestures using a combination of Artificial Neural Networks (ANN) and the MediaPipe framework. The goal is to facilitate seamless communication between individuals who use ASL and those who don’t understand it, by converting hand signs into readable text instantly.

Here is the demonstration of the model built:

https://github.com/user-attachments/assets/9bc1ed97-12ba-48c3-9fb2-f5ddda8dfe9f


## Overview

American Sign Language is a vital communication method for the deaf and hard-of-hearing community. However, many people are not familiar with ASL, which can create barriers in everyday conversations. This project addresses that gap by using computer vision and machine learning to recognize and translate ASL signs in real-time.

The system captures video from a standard webcam and uses MediaPipe to track the position of the hand and fingers by detecting specific landmarks on the hand. These landmarks are then processed by a trained Artificial Neural Network, which classifies the input into corresponding ASL alphabets or gestures. The recognized sign is displayed as text on the screen, providing instant feedback.

## Key Features

- **Real-Time Hand Gesture Recognition**  
  The system processes video input continuously to detect and interpret ASL gestures live, making interactions fluid and natural.

- **MediaPipe Hand Tracking**  
  MediaPipe is a powerful framework for hand detection and tracking. It identifies key points on the hand, such as fingertips and joints, which are crucial for accurate gesture recognition.

- **Artificial Neural Network (ANN) Classifier**  
  The ANN model is trained on various ASL signs and learns to distinguish between them based on the hand landmarks. This enables the system to correctly predict the signs users make.

- **User-Friendly Interface**  
  The program displays the detected sign as text, making it easy to understand and use. It requires only a standard webcam and runs on most modern computers.

## How It Works

1. **Capture**: The webcam captures live video frames.  
2. **Detect**: MediaPipe analyzes each frame to locate the hand and its key landmarks.  
3. **Predict**: The extracted landmark data is fed into the ANN model, which predicts the most likely ASL sign.  
4. **Display**: The recognized sign is shown on the screen, updating as the user changes their hand gestures.

## Why This Matters

- **Accessibility**: Provides an assistive tool that bridges communication between the deaf community and non-signers.  
- **Education**: Helps learners practice and verify ASL gestures in real-time.  
- **Innovation**: Combines modern computer vision with machine learning to solve real-world problems.

## Getting Started

### Prerequisites

- A computer with a webcam.  
- Python 3.12.3 installed.  
- Basic Python packages like OpenCV, MediaPipe, and TensorFlow/PyTorch for the ANN model.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nitheesh1904/American_sign_language.git
   cd code 
   ```
2. Create a virtual environment and activate it:

   ```bash   
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
    ```bash
     pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
     python3 my-app.py
    ```
