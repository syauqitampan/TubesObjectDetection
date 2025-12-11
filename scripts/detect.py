from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("../models/yolo11x.pt")

allowed_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

source = "videos/test.mp4"      # atau "images/test.jpg"

# ---- Cek apakah input adalah gambar ----
image_ext = [".jpg", ".jpeg", ".png", ".bmp"]
video_ext = [".mp4", ".avi", ".mov", ".mkv"]

ext = os.path.splitext(source)[1].lower()

# ---------------------------
# MODE: IMAGE
# ---------------------------
if ext in image_ext:
    img = cv2.imread(source)
    results = model(img)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Hasil Deteksi Gambar", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------
# MODE: VIDEO
# ---------------------------
elif ext in video_ext:
    cap = cv2.VideoCapture(source)
    cv2.namedWindow("Deteksi", cv2.WINDOW_NORMAL)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in allowed_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        cv2.imshow("Deteksi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Format file tidak dikenali! Gunakan gambar atau video.")