# Blind Assist — AI Assistance for the Visually Impaired

## About
Blind Assist is a prototype that combines AI-based object detection, text recognition, voice announcements, and ultrasonic obstacle alerts to assist visually impaired people.

## Features
- Real-time object detection using YOLO.
- Left, right, and ahead direction announcements.
- Text recognition using EasyOCR.
- Natural voice announcements using Google Text-to-Speech.
- Audio streaming to Raspberry Pi and Bluetooth earbuds.
- Ultrasonic obstacle alerts using an HC-SR04 sensor.
- Optional face recognition using OpenCV.

## Technologies
Python, OpenCV, YOLO, EasyOCR, PyTorch, gTTS, Raspberry Pi, and HC-SR04.

## System Architecture
- **Windows laptop:** Performs camera processing, object detection, OCR, and speech generation.
- **Raspberry Pi:** Receives audio and handles ultrasonic obstacle alerts.

**Note:** AI processing currently runs on the laptop. A fully standalone Raspberry Pi implementation is future work.

## Safety
This is a prototype, not a certified assistive device. Object detection and ultrasonic alerts cannot guarantee collision avoidance. It should not replace a mobility cane or guide dog.

## Future Improvements
- Run AI inference directly on Raspberry Pi.
- Test the Raspberry Pi camera.
- Improve offline speech and obstacle detection.