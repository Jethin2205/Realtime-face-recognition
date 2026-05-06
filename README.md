# Realtime-face-recognition
# Face Recognition System with Age, Gender & Emotion Detection

Developed a real-time face recognition system using OpenCV to identify individuals and predict age, gender, and emotion. Integrated SQLite for persistent data storage and activity logging, enabling full audit trails for recognized individuals. Applied deep learning models for multi-attribute facial analysis, enhancing system accuracy.

---

## Features

- Real-time face detection via webcam
- Face recognition — identifies registered individuals by name
- Predicts Age, Gender and Emotion for every detected face
- SQLite database stores all face encodings and detection logs
- Full audit trail — every detection is timestamped and saved
- Live face registration (press R while running)
- Register from a photo using command line
- Export logs to CSV anytime

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| OpenCV | Webcam capture, face detection, UI rendering |
| face-recognition / dlib | 128-d face encoding and matching |
| Caffe / ONNX Models | Age, gender, emotion prediction |
| SQLite | Persistent storage for persons and logs |
| NumPy | Numerical operations on face vectors |

---

## Project Structure

```
face_recognition_system/
│
├── main.py                     # Entry point, runs the webcam loop
├── database.py                 # SQLite operations
├── face_recognition_module.py  # Face detection and recognition
├── emotion_detection.py        # Age, gender, emotion prediction
├── utils.py                    # Drawing, HUD, registration helpers
│
├── models/                     # Place pre-trained model files here
├── data/
│   ├── face_recognition.db     # Auto-created SQLite database
│   └── logs_export.csv         # Generated when you press E
│
└── requirements.txt
```

---

## Database Schema

```sql
CREATE TABLE persons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    age           TEXT,
    gender        TEXT,
    face_encoding BLOB NOT NULL,
    registered_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE logs (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER,
    timestamp  TEXT NOT NULL,
    emotion    TEXT,
    confidence REAL,
    FOREIGN KEY (person_id) REFERENCES persons(id)
);
```

---

## Installation

**Step 1 — Clone the project**
```bash
git clone https://github.com/Jethin2205/face-recognition-system
cd face_recognition_system
```

**Step 2 — Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate       # Mac / Linux
venv\Scripts\activate          # Windows
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

> **Windows users:** If dlib install fails, download a pre-built wheel from
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x
> then run: `pip install path/to/dlib-*.whl`

---

## How to Run

**Start the system:**
```bash
python main.py
```

**Custom camera:**
```bash
python main.py --camera 1
```

**Register from a photo:**
```bash
python main.py --register-from-image photos/john.jpg
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| `R` | Register new face from webcam |
| `Q` | Quit |
| `E` | Export detection logs to CSV |
| `S` | Print database stats in terminal |

---

## How Registration Works

**Live webcam:**
1. Run `python main.py`
2. Press `R`
3. 3-second countdown — look at the camera
4. Enter name, age, gender in the terminal
5. Face saved to database — recognized from next frame onward

**From a photo:**
```bash
python main.py --register-from-image photo.jpg
```

---

## Viewing the Database

Download **DB Browser for SQLite** → https://sqlitebrowser.org

Open: `data/face_recognition.db`

You will see all registered persons and detection logs in a spreadsheet view.

---

## Optional Models for Better Accuracy

Place these files in the `models/` folder:

| File | Purpose |
|---|---|
| `age_deploy.prototxt` + `age_net.caffemodel` | Age prediction |
| `gender_deploy.prototxt` + `gender_net.caffemodel` | Gender prediction |
| `emotion_mini_xception.onnx` | Emotion detection |

Without these files the system still runs using built-in heuristic fallback.

Download links:
- Age/Gender models: https://github.com/spmallick/learnopencv/tree/master/AgeGender
- Emotion ONNX model: https://github.com/oarriaga/face_classification

---

## How It Works Internally

1. Webcam frame is captured by OpenCV
2. Haarcascade detects face bounding boxes
3. dlib computes a 128-number vector for each face
4. Vector is compared against all stored encodings in SQLite
5. Closest match below the tolerance threshold → name displayed
6. Face crop is passed to age, gender and emotion models
7. Detection is logged to the database with timestamp

---

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
face-recognition>=1.3.0
dlib>=19.24.0
Pillow>=9.0.0
```
