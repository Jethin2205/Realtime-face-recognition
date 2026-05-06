import cv2, numpy as np, face_recognition
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DetectedFace:
    top: int; right: int; bottom: int; left: int
    name: str = "Unknown"
    person_id: int = None
    confidence: float = 0.0
    encoding: np.ndarray = field(default=None, repr=False)

    @property
    def bbox(self):
        return self.left, self.top, self.right-self.left, self.bottom-self.top


class FaceDetector:
    CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(self.CASCADE)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
        return [(y, x+w, y+h, x) for (x,y,w,h) in rects] if len(rects) else []


class FaceRecognizer:
    TOLERANCE = 0.55

    def __init__(self, tolerance=None):
        if tolerance: self.TOLERANCE = tolerance
        self._encodings, self._names, self._ids = [], [], []

    def load(self, persons):
        self._encodings = [p["face_encoding"] for p in persons]
        self._names     = [p["name"] for p in persons]
        self._ids       = [p["id"] for p in persons]
        print(f"Loaded {len(persons)} known face(s).")

    def recognize(self, frame, locations):
        if not locations: return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
        results = []
        for loc, enc in zip(locations, encodings):
            top, right, bottom, left = loc
            face = DetectedFace(top=top, right=right, bottom=bottom, left=left, encoding=enc)
            if self._encodings and enc is not None:
                dists = face_recognition.face_distance(self._encodings, enc)
                idx = int(np.argmin(dists))
                if dists[idx] <= self.TOLERANCE:
                    face.name, face.person_id = self._names[idx], self._ids[idx]
                    face.confidence = round(1.0 - float(dists[idx]), 3)
            results.append(face)
        return results

    @staticmethod
    def encode_from_image(path):
        img = face_recognition.load_image_file(str(path))
        encs = face_recognition.face_encodings(img)
        return encs[0] if encs else None

    @staticmethod
    def encode_from_frame(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
        return encs[0] if encs else None
