import streamlit as st
from ultralytics import YOLO
from collections import defaultdict
object_ids = defaultdict(set)
import cv2
import numpy as np

st.title("Deteksi Kendaraan (Gambar & Video)")

model_option = st.selectbox("Pilih model", ["yolo11s.pt", "yolo11n.pt", "yolo11x.pt"])
model = YOLO(f"models/{model_option}")

TRAFFIC_THRESHOLD = 20

class_map = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

allowed_classes = [2, 3, 5, 7]

mode = st.radio("Pilih mode input:", ["Gambar", "Video"])

# ==========================
# MODE GAMBAR
# ==========================
if mode == "Gambar":
    img_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Hitung jumlah kendaraan per kelas
        counts = defaultdict(int)

        results = model(img)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in allowed_classes:
                    continue

                counts[cls] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls]
                conf = float(box.conf[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f"{label} {conf:.2f}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,0), 2)

        st.image(img, channels="BGR")
        
        total_kendaraan = sum(counts.values())

        if total_kendaraan >= TRAFFIC_THRESHOLD:
            st.error(f"RAMAI 🚗🚕 ({total_kendaraan} kendaraan)")
        else:
            st.success(f"SEPI 🛣️ ({total_kendaraan} kendaraan)")

        # Tampilkan jumlah kendaraan
        st.subheader("📊 Jumlah Kendaraan (Gambar)")
        class_map = {
            2: "Mobil",
            3: "Motor",
            5: "Bus",
            7: "Truk"
        }
        for cls, total in counts.items():
            st.write(f"{class_map[cls]} : {total}")
            

# ==========================
# MODE VIDEO
# ==========================
else:
    vid_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    if vid_file is not None:
        # Simpan file sementara
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(vid_file.read())

        # --- LOGIKA TOMBOL STOP/START ---
        # Inisialisasi state 'run' jika belum ada
        if 'run' not in st.session_state:
            st.session_state.run = False

        # Buat kolom agar tombol berdampingan
        col1, col2 = st.columns(2)
        
        with col1:
            # Tombol Start
            if st.button("Mulai Deteksi"):
                st.session_state.run = True
                object_ids.clear()   # ⬅️ PENTING

        
        with col2:
            # Tombol Stop
            if st.button("Stop"):
                st.session_state.run = False

        # Placeholder untuk menampilkan video
        stframe = st.empty()

        # Hanya jalankan loop jika status 'run' adalah True
        if st.session_state.run:
            cap = cv2.VideoCapture(tfile)
            
            while cap.isOpened():
                ret, frame = cap.read()
                # Jika video habis atau tombol Stop ditekan (state berubah jadi False)
                if not ret or not st.session_state.run:
                    break

                results = model.track(frame, persist=True, imgsz=640, conf=0.4)

                for r in results:
                    if r.boxes.id is None:
                        continue

                    for box, track_id in zip(r.boxes, r.boxes.id):
                        cls = int(box.cls[0])
                        if cls not in allowed_classes:
                            continue

                        object_ids[cls].add(int(track_id))

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names[cls]
                        conf = float(box.conf[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0,255,0), 2)

                stframe.image(frame, channels="BGR")
                
                total_kendaraan = sum(len(ids) for ids in object_ids.values())

                # Hapus tampilan per frame, pindahkan ke akhir
            
            cap.release()
            
            total_kendaraan = sum(len(ids) for ids in object_ids.values())

            if total_kendaraan >= TRAFFIC_THRESHOLD:
                st.error(f"RAMAI 🚗🚕 ({total_kendaraan} kendaraan)")
            else:
                st.success(f"SEPI 🛣️ ({total_kendaraan} kendaraan)")
            
            col1, col2, col3, col4 = st.columns(4)

            for cls, ids in object_ids.items():
                name = model.names[cls]
                if name == "car":
                    col1.metric("Mobil", len(ids))
                elif name == "motorcycle":
                    col2.metric("Motor", len(ids))
                elif name == "bus":
                    col3.metric("Bus", len(ids))
                elif name == "truck":
                    col4.metric("Truk", len(ids))

            
            # Reset state setelah video selesai/stop agar tombol Mulai bisa ditekan lagi
            if st.session_state.run:
                st.session_state.run = False
                st.success("Video Selesai!")
            else:
                st.warning("Video Dihentikan.")