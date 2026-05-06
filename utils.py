import cv2, numpy as np, face_recognition

FONT = cv2.FONT_HERSHEY_SIMPLEX

EMOTION_COLORS = {
    "Happy": (0,230,50), "Sad": (200,60,60), "Angry": (0,0,255),
    "Neutral": (180,180,180), "Surprise": (0,200,240),
    "Fear": (140,60,200), "Disgust": (60,140,60)
}

def put_text_bg(frame, text, pos, scale=0.55, fg=(255,255,255), bg=(20,20,20)):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    x, y = pos
    cv2.rectangle(frame, (x-3, y-th-3), (x+tw+3, y+bl+3), bg, cv2.FILLED)
    cv2.putText(frame, text, (x, y), FONT, scale, fg, 1, cv2.LINE_AA)

def draw_faces(frame, faces, analyses):
    out = frame.copy()
    for face, a in zip(faces, analyses):
        known = face.name != "Unknown"
        color = (0,220,100) if known else (0,80,200)
        emo_c = EMOTION_COLORS.get(a.emotion, (200,200,200))

        cv2.rectangle(out, (face.left, face.top), (face.right, face.bottom), color, 2)

        label = f"{face.name}  {face.confidence:.0%}" if known else "Unknown"
        put_text_bg(out, label,                    (face.left, face.top - 8),   fg=color)
        put_text_bg(out, f"Age: {face.age if hasattr(face,'age') else a.age}",   (face.left, face.bottom+18), fg=(200,230,255))
        put_text_bg(out, a.gender,                 (face.left+90, face.bottom+18), fg=(255,210,180))
        put_text_bg(out, f"{a.emotion} {a.emotion_conf:.0%}", (face.left, face.bottom+36), fg=emo_c)
    return out

def draw_hud(frame, n_known, n_unknown, fps, total):
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (220,80), (15,15,15), cv2.FILLED)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    for i, txt in enumerate([f"FPS: {fps:.1f}", f"Known: {n_known}",
                              f"Unknown: {n_unknown}", f"Registered: {total}"]):
        cv2.putText(frame, txt, (8, 16+i*16), FONT, 0.45, (200,220,200), 1, cv2.LINE_AA)
    h = frame.shape[0]
    cv2.putText(frame, "[R]Register [Q]Quit [E]Export", (6, h-8), FONT, 0.38, (120,120,120), 1)
    return frame

def register_live(cap, db, recognizer, analyser):
    print("Look at the camera — capturing in 3 seconds...")
    for c in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret: return False
        cv2.putText(frame, str(c), (frame.shape[1]//2-30, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_DUPLEX, 4.0, (0,230,80), 6)
        cv2.imshow("Face Recognition System", frame)
        cv2.waitKey(1000)

    ret, frame = cap.read()
    if not ret: return False

    encoding = recognizer.encode_from_frame(frame)
    if encoding is None:
        print("No face detected — cancelled."); return False

    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    a    = analyser.analyse(frame[locs[0][0]:locs[0][2], locs[0][3]:locs[0][1]]) if locs else None

    print(f"Auto → Age: {a.age if a else '?'}  Gender: {a.gender if a else '?'}")
    name = input("Name: ").strip()
    if not name: print("Cancelled."); return False
    age  = input(f"Age (Enter={a.age if a else '?'}): ").strip() or (a.age if a else None)
    gen  = input(f"Gender (Enter={a.gender if a else '?'}): ").strip() or (a.gender if a else None)

    pid = db.add_person(name, age, gen, encoding)
    recognizer.load(db.get_all_persons())
    print(f"Registered '{name}' (id={pid})")
    return True

def crop(frame, face):
    h, w = frame.shape[:2]
    return frame[max(0,face.top):min(h,face.bottom), max(0,face.left):min(w,face.right)]

def fps_calc(prev):
    curr = cv2.getTickCount()
    return cv2.getTickFrequency() / (curr - prev + 1e-9), curr
