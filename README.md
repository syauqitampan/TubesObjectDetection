# Tubes Object Detection

Aplikasi ini adalah tugas besar mata kuliah Computer Vision untuk melakukan deteksi objek menggunakan algoritma **YOLOv11**.

Muhammad Syauqi Alghifari 4.33.24.0.19

## ⚙️ Fitur
* Mendeteksi objek Kendaraan dari video & gambar.
* Menggunakan model pre-trained YOLO11x.
* (Tulis fitur lain projectmu di sini, misal: Tracking, Counting, dll)

## 🛠️ Instalasi

Karena file model dan library tidak disertakan dalam repository ini (untuk menghemat penyimpanan), silakan ikuti langkah berikut untuk menjalankannya:

### 1. Clone Repository
```bash
git clone [https://github.com/syauqitampan/TubesObjectDetection.git](https://github.com/syauqitampan/TubesObjectDetection.git)
cd TubesObjectDetection

# Untuk Windows
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

dowloan model yolo11x.pt dan masukkan kedalam folder models/ (bisa menggunakan yolo versi yang lain)
