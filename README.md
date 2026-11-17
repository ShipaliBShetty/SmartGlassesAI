PROJECT TITLE:
Agentic AI-Based Social Interaction Assistant for Visually Impaired

TEAM DETAILS:

1. Rhea Wilson Fernandes – 4CB23AI080
2. Shipali B Shetty – 4CB23AI096

PROJECT DESCRIPTION:

SmartGlassesAI is an AI-based assistive system designed to help visually impaired individuals understand social interactions through face recognition, emotion detection, gesture recognition and audio feedback.

SOFTWARE REQUIREMENTS:

1. Python – https://www.python.org/
2. Visual Studio Code – https://code.visualstudio.com/
3. Arduino IDE – https://www.arduino.cc/en/software
4. OpenCV – https://opencv.org/
5. InsightFace – https://github.com/deepinsight/insightface
6. DeepFace – https://github.com/serengil/deepface
7. MediaPipe – https://ai.google.dev/edge/mediapipe/solutions/guide
8. ONNX Runtime – https://onnxruntime.ai/
9. SQLite – https://www.sqlite.org/
10. pyttsx3 – https://pypi.org/project/pyttsx3/

DATASET USED:

The project uses enrolled facial information for face recognition. InsightFace generates 512-dimensional facial embeddings, which are associated with names and stored in an SQLite database.

DATASET GENERATED:

The SQLite database (faces.db) is generated during the enrollment process. It stores the names and facial embeddings of enrolled individuals. The database can be created and populated on the college computer through the enrollment module.

HARDWARE REQUIREMENTS:

1. ESP32-CAM with OV3660 camera
2. ESP32-CAM-MB programmer
3. USB cable
4. Earphones
5. Laptop or computer

HOW TO RUN THE PROJECT:

1. Install Python and the required software.
2. Create and activate a Python virtual environment.
3. Install dependencies using requirements.txt.
4. Configure the ESP32-CAM Wi-Fi connection and camera stream URL.
5. Run the application using: python main.py

PROJECT FOLDER:
SmartGlassesAI
