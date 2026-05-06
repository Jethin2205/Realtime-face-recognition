import cv2, sys, time, argparse
from pathlib import Path

from database               import DatabaseManager
from face_recognition_module import FaceDetector, FaceRecognizer
from emotion_detection       import FaceAnalyser
from utils                   import draw_faces, draw_hud, register_live, crop, fps_calc

for d in ("models", "data"):
    Path(d).mkdir(exist_ok=True)


class FaceSystem:
    SKIP = 2          # process every N-th frame
    LOG_INTERVAL = 5  # seconds between DB log writes per person

    def __init__(self, args):
        self.args      = args
        self.db        = DatabaseManager()
        self.detector  = FaceDetector()
        self.recognizer = FaceRecognizer(tolerance=args.tolerance)
        self.analyser  = FaceAnalyser()

        self.recognizer.load(self.db.get_all_persons())

        self._frame   = 0
        self._fps     = 0.0
        self._tick    = cv2.getTickCount()
        self._faces   = []
        self._analyses = []
        self._logged  = {}

    def run(self):
        cap = cv2.VideoCapture(self.args.camera)
        if not cap.isOpened():
            print(f"Cannot open camera {self.args.camera}"); sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Running — Q=Quit  R=Register  E=Export CSV")

        while True:
            ret, frame = cap.read()
            if not ret: break

            self._frame += 1
            self._fps, self._tick = fps_calc(self._tick)

            if self._frame % self.SKIP == 0:
                scale = self.args.scale
                small = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
                locs  = self.detector.detect(small)
                locs  = [(int(t/scale),int(r/scale),int(b/scale),int(l/scale)) for t,r,b,l in locs]

                self._faces    = self.recognizer.recognize(frame, locs)
                self._analyses = [self.analyser.analyse(crop(frame, f)) for f in self._faces]
                self._log()

            display = draw_faces(frame, self._faces, self._analyses)
            s = self.db.stats()
            nk = sum(1 for f in self._faces if f.name != "Unknown")
            display = draw_hud(display, nk, len(self._faces)-nk, self._fps, s["persons"])
            cv2.imshow("Face Recognition System", display)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("r"): register_live(cap, self.db, self.recognizer, self.analyser)
            elif key == ord("e"): print("Exported →", self.db.export_csv())
            elif key == ord("s"): print(self.db.stats())

        cap.release(); cv2.destroyAllWindows()

    def _log(self):
        now = time.time()
        for face, a in zip(self._faces, self._analyses):
            key = face.person_id
            if now - self._logged.get(key, 0) >= self.LOG_INTERVAL:
                self.db.add_log(face.person_id, a.emotion, a.emotion_conf)
                self._logged[key] = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera",    type=int,   default=0)
    ap.add_argument("--scale",     type=float, default=0.5)
    ap.add_argument("--tolerance", type=float, default=0.55)
    ap.add_argument("--register-from-image", type=str, default=None)
    args = ap.parse_args()

    if args.register_from_image:
        db  = DatabaseManager()
        rec = FaceRecognizer(args.tolerance)
        rec.load(db.get_all_persons())
        enc = FaceRecognizer.encode_from_image(args.register_from_image)
        if enc is None: print("No face found in image."); return
        name = input("Name: ").strip()
        age  = input("Age: ").strip()
        gen  = input("Gender: ").strip()
        pid  = db.add_person(name, age, gen, enc)
        print(f"Registered '{name}' (id={pid})")
        return

    FaceSystem(args).run()

if __name__ == "__main__":
    main()
