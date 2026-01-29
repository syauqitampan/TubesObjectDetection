from ultralytics import YOLO
from collections import defaultdict
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

# ==========================
# MODE GAMBAR
# ==========================
if mode == "Gambar":
    img_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img, conf=0.4)

        counter = defaultdict(int)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in allowed_classes:
                    continue

                counter[cls] += 1   # ⬅️ HITUNG JUMLAH

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f"{label} {conf:.2f}",
                            (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        # Tampilkan gambar
        st.image(img, channels="BGR")

        # ==========================
        # TAMPILKAN JUMLAH
        # ==========================
        st.subheader("📊 Jumlah Kendaraan")

        col1, col2, col3, col4 = st.columns(4)

        col_map = {
            "car": col1,
            "motorcycle": col2,
            "bus": col3,
            "truck": col4
        }

        for cls, count in counter.items():
            name = model.names[cls]
            if name in col_map:
                col_map[name].metric(name.capitalize(), count)


# ---------------------------
# MODE: VIDEO
# ---------------------------
elif ext in video_ext:
    cap = cv2.VideoCapture(source)
    cv2.namedWindow("Deteksi", cv2.WINDOW_NORMAL)

    object_ids = defaultdict(set)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, conf=0.4)

        for r in results:
            if r.boxes.id is None:
                continue

            for box, track_id in zip(r.boxes, r.boxes.id):
                cls = int(box.cls[0])
                if cls not in allowed_classes:
                    continue

                track_id = int(track_id.item())
                object_ids[cls].add(track_id)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f} ID:{track_id}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)


        cv2.imshow("Deteksi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
print("\n===== HASIL TOTAL KENDARAAN =====")
for cls, ids in object_ids.items():
    print(f"{model.names[cls]:12s}: {len(ids)}")


else:
    print("Format file tidak dikenali! Gunakan gambar atau video.")