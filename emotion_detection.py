import cv2, numpy as np
from dataclasses import dataclass
from pathlib import Path

MODEL_DIR    = Path("models")
AGE_LIST     = ["(0-2)","(4-6)","(8-12)","(15-20)","(25-32)","(38-43)","(48-53)","(60-100)"]
GENDER_LIST  = ["Male","Female"]
EMOTION_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
MEAN_VALS    = (78.4263377603, 87.7689143744, 114.895847746)

@dataclass
class Result:
    age: str = "N/A";      age_conf: float = 0.0
    gender: str = "N/A";   gender_conf: float = 0.0
    emotion: str = "Neutral"; emotion_conf: float = 0.0


class EmotionDetector:
    SIZE = (64, 64)

    def __init__(self):
        self.net, self.mode = None, "heuristic"
        onnx = MODEL_DIR / "emotion_mini_xception.onnx"
        if onnx.exists():
            try:
                self.net  = cv2.dnn.readNetFromONNX(str(onnx))
                self.mode = "onnx"; print("Emotion model loaded.")
            except: pass

    def predict(self, face_bgr):
        if self.mode == "onnx" and self.net:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            blob = cv2.resize(gray, self.SIZE).astype("float32") / 255.0
            self.net.setInput(blob.reshape(1,1,*self.SIZE))
            probs = self._softmax(self.net.forward().flatten())
            idx = int(np.argmax(probs))
            return EMOTION_LIST[idx], float(probs[idx])
        return self._heuristic(face_bgr)

    def _heuristic(self, face):
        g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        m, s = float(np.mean(g)), float(np.std(g))
        if m > 140 and s > 50: return "Happy", 0.55
        if m < 80:             return "Sad",   0.50
        if s > 70:             return "Surprise", 0.48
        return "Neutral", 0.60

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x)); return e / e.sum()


class AgeGenderDetector:
    def __init__(self):
        self.age_net = self.gen_net = None
        ap = MODEL_DIR/"age_deploy.prototxt";    am = MODEL_DIR/"age_net.caffemodel"
        gp = MODEL_DIR/"gender_deploy.prototxt"; gm = MODEL_DIR/"gender_net.caffemodel"
        if all(p.exists() for p in [ap, am, gp, gm]):
            self.age_net = cv2.dnn.readNet(str(am), str(ap))
            self.gen_net = cv2.dnn.readNet(str(gm), str(gp))
            print("Age/Gender models loaded.")
        else:
            print("Age/Gender models missing → using heuristic. Place .caffemodel files in models/")

    def predict(self, face):
        if self.age_net:
            blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MEAN_VALS)
            self.gen_net.setInput(blob); gp = self.gen_net.forward().flatten()
            self.age_net.setInput(blob); ap = self.age_net.forward().flatten()
            return AGE_LIST[np.argmax(ap)], float(ap.max()), GENDER_LIST[np.argmax(gp)], float(gp.max())
        g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gender = "Male" if float(np.mean(face[:,:,2])) > 140 else "Female"
        age    = "(25-32)" if float(np.mean(g)) > 100 else "(38-43)"
        return age, 0.5, gender, 0.5


class FaceAnalyser:
    def __init__(self):
        self.ag = AgeGenderDetector()
        self.em = EmotionDetector()

    def analyse(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0: return Result()
        age, ac, gender, gc = self.ag.predict(face_bgr)
        emotion, ec         = self.em.predict(face_bgr)
        return Result(age=age, age_conf=ac, gender=gender,
                      gender_conf=gc, emotion=emotion, emotion_conf=ec)
