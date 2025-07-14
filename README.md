
# Facial Emotion Recognition

This project implements facial emotion recognition using a custom-trained Convolutional Neural Network (CNN) and real-time webcam inference with OpenCV. It also provides an alternative approach using DeepFace.

## Features

- Train a CNN on grayscale face images for emotion classification.
- Real-time emotion recognition from webcam video using OpenCV.
- Optionally, use DeepFace for emotion analysis.

## Project Structure

- `.idea/cnn_e.py` — Train and save the CNN emotion recognition model.
- `.idea/test_cnn_e.py` — Real-time emotion recognition using the trained model and OpenCV.
- `.idea/deepface_e.py` — Real-time emotion recognition using DeepFace.

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- DeepFace (optional)
- Numpy

Install dependencies:
```
pip install tensorflow keras opencv-python deepface numpy
```

## Usage

1. **Train the Model:**
   - Place your training and test images in the appropriate folders.
   - Run `.idea/cnn_e.py` to train and save the model (`emotion_model2.h5`).

2. **Real-Time Recognition:**
   - Run `.idea/test_cnn_e.py` to use your webcam for live emotion detection.

3. **DeepFace Alternative:**
   - Run `.idea/deepface_e.py` for emotion recognition using DeepFace.

## Dataset

- The model expects grayscale face images of size 48x48, organized in subfolders by emotion class.
- Example classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## License

This project is for educational purposes.

---
