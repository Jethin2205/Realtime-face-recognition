import cv2, numpy as np
from dataclasses import dataclass

@dataclass
class Result:
    age: str = "N/A";          age_conf: float = 0.0
    gender: str = "N/A";       gender_conf: float = 0.0
    emotion: str = "Neutral";  emotion_conf: float = 0.0


class FaceAnalyser:
    def __init__(self):
        self._deepface = None
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            print("DeepFace loaded — Age, Gender, Emotion ready.")
        except:
            print("DeepFace not found. Run: pip install deepface")

    def analyse(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return Result()

        if self._deepface:
            try:
                result = self._deepface.analyze(
                    face_bgr,
                    actions=["age", "gender", "emotion"],
                    enforce_detection=False,
                    silent=True
                )
                r = result[0]

                age     = str(r.get("age", "N/A"))
                gender  = r.get("dominant_gender", "N/A").capitalize()
                emotion = r.get("dominant_emotion", "Neutral").capitalize()

                emotion_conf = r["emotion"][r["dominant_emotion"]] / 100.0
                gender_conf  = r["gender"][r["dominant_gender"]] / 100.0

                return Result(
                    age=f"({age})",
                    age_conf=0.85,
                    gender=gender,
                    gender_conf=round(gender_conf, 2),
                    emotion=emotion,
                    emotion_conf=round(emotion_conf, 2)
                )
            except Exception as e:
                pass

        return self._heuristic(face_bgr)

    def _heuristic(self, face):
        g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        m, s = float(np.mean(g)), float(np.std(g))
        emotion = "Happy" if (m > 140 and s > 50) else "Sad" if m < 80 else "Neutral"
        gender  = "Male" if float(np.mean(face[:,:,2])) > 140 else "Female"
        age     = "(25-32)" if m > 100 else "(38-43)"
        return Result(age=age, gender=gender, emotion=emotion,
                      age_conf=0.5, gender_conf=0.5, emotion_conf=0.5)
