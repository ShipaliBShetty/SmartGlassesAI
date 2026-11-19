# SMARTGLASSESAI

## Project Title
Agentic AI-Based Social Interaction Assistant for Visually Impaired

## Team Members
1. Rhea Wilson Fernandes – 4CB23AI080
2. Shipali B Shetty – 4CB23AI096

## Project Description
SmartGlassesAI is an AI-based assistive system designed to help visually impaired individuals understand social interactions. The system uses a camera mounted on smart glasses to capture the surroundings and applies artificial intelligence techniques to recognize people, analyze facial expressions, and identify gestures.

The processed information is selected by a decision-making module and converted into audio feedback through earphones.


## Technologies Used

### Hardware
- ESP32-CAM with OV3660 camera
- ESP32-CAM-MB programmer
- USB cable
- Earphones
- Smart glasses frame

### Software
- Python
- Arduino IDE
- OpenCV
- InsightFace
- DeepFace
- MediaPipe
- ONNX Runtime
- SQLite
- pyttsx3
- VS Code


## DATASET USED:

The project uses enrolled facial information for face recognition. InsightFace generates 512-dimensional facial embeddings, which are associated with names and stored in an SQLite database.

## DATASET GENERATED:

The SQLite database (faces.db) is generated during the enrollment process. It stores the names and facial embeddings of enrolled individuals. The database can be created and populated on the college computer through the enrollment module.


## Project Modules
1. Face Recognition – InsightFace
2. Emotion Detection – DeepFace
3. Gesture Recognition – MediaPipe
4. Database Management – SQLite
5. Decision-Making Module
6. Audio Feedback – pyttsx3
7. Camera Streaming – ESP32-CAM

## Installation Requirements

1. Install Python 3.11.
2. Install Visual Studio Code.
3. Install Arduino IDE.
4. Install the ESP32 board package in Arduino IDE.
5. Install the required Python packages using:

   pip install -r requirements.txt

## How to Run the Project

### Step 1: Camera Setup
- Open the Arduino folder.
- Open esp32_cam.ino using Arduino IDE.
- Configure the Wi-Fi credentials.
- Select the appropriate ESP32 board and COM port.
- Upload the code to the ESP32-CAM.

### Step 2: Python Setup
- Open the Python folder in VS Code.
- Create and activate a virtual environment.
- Install the required packages.
- Update the camera stream address in config.py.

### Step 3: Run the Application

   python main.py

## Dataset Information
The system uses facial information enrolled by the user. InsightFace generates facial embeddings, which are stored in the SQLite database for recognizing previously enrolled individuals.

No external training dataset is required for the face enrollment database. Emotion and gesture recognition use their respective pre-trained models.

## Important Notes
- The ESP32-CAM and laptop should be connected to the appropriate Wi-Fi network for camera streaming.
- Update the camera IP address whenever the network changes.
- Enroll people before testing face recognition.
- Do not upload real biometric records or Wi-Fi passwords to a public repository.

## Project Objective
To provide visually impaired individuals with useful social information through real-time computer vision and audio feedback.

PROJECT FOLDER:
SmartGlassesAI
