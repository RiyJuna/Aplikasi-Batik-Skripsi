import numpy as np
import os
import gradio as gr

# Tambahkan import Flask yang diperlukan
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ======================================================
# INISIALISASI FLASK
# ======================================================
app = Flask(__name__)

# ======================================================
# LABEL KELAS
# ======================================================
dic = {
    0: 'Aceh_Pintu_Aceh',
    1: 'Bali_Barong',
    2: 'Bali_Merak',
    3: 'DKI_Ondel_Ondel',
    4: 'JawaBarat_Megamendung',
    5: 'JawaTimur_Pring',
    6: 'Kalimantan_Dayak',
    7: 'Lampung_Gajah',
    8: 'Madura_Mataketeran',
    9: 'Maluku_Pala',
    10: 'NTB_Lumbung',
    11: 'Papua_Asmat',
    12: 'Papua_Cendrawasih',
    13: 'Papua_Tifa',
    14: 'Solo_Parang',
    15: 'SulawesiSelatan_Lontara',
    16: 'SumateraBarat_Rumah_Minang',
    17: 'SumateraUtara_Boraspati',
    18: 'Yogyakarta_Kawung',
    19: 'Yogyakarta_Parang'
}

# ======================================================
# LOAD MODEL
# ======================================================
model = load_model('MobileNetV2-Batik v1-72.00.h5')

# ======================================================
# PREDIKSI FUNCTION (Untuk Gradio dan Flask)
# ======================================================
def predict(img):
    # Jika input berupa path string (dari Flask upload)
    if isinstance(img, str):
        img = image.load_img(img, target_size=(224, 224))
    else:
        # Jika dari Gradio (sudah berupa PIL Image)
        img = img.resize((224, 224))
        
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    confidence = np.max(pred)
    
    # Mengembalikan label indeks, teks nama kelas, dan nilai confidence
    class_idx = np.argmax(pred)
    class_name = dic[class_idx]
    
    return class_name, confidence

# Fungsi pembungkus khusus untuk output teks Gradio
def predict_gradio(img):
    class_name, confidence = predict(img)
    if confidence < 0.70:
        return "Bukan Motif Batik"
    return f"{class_name} ({round(confidence*100,2)}%)"

# ======================================================
# UI GRADIO CONFIGURATION
# ======================================================
demo = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Klasifikasi Batik Nusantara"
)

# Integrasikan Gradio ke Flask dengan rute /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

# ======================================================
# ROUTE FLASK: HALAMAN UTAMA / DATA BATIK
# ======================================================
@app.route("/")
def main():
    data_batik = [
        {
            "nama": "Aceh - Pintu Aceh",
            "pulau" : "Sumatera",
            "gambar": [
                "/static/batik/Aceh Pintu Aceh/aceh1.jpg",
                "/static/batik/Aceh Pintu Aceh/aceh2.jpg",
                "/static/batik/Aceh Pintu Aceh/aceh3.jpg"
            ],
            "filosofi": "Melambangkan karakter masyarakat yang terbuka dalam menerima perbedaan namun tetap menjaga batas privasi..."
        },
        {
            "nama": "Bali - Barong",
            "pulau" : "bali_ntt_ntb",
            "gambar": [
                "/static/batik/Bali Barong/1.jpg",
                "/static/batik/Bali Barong/2.jpg",
                "/static/batik/Bali Barong/3.jpg"
            ],
            "filosofi": "Merepresentasikan filosofi Rwa Bhineda, yakni keseimbangan abadi antara kebaikan dan kejahatan di dunia..."
        },
        # ... (Data batik lainnya tetap biarkan seperti kodemu sebelumnya)
    ]

    return render_template("data_batik.html", data_batik=data_batik)

# ======================================================
# ROUTE FLASK: HALAMAN CLASSIFICATION
# ======================================================
@app.route("/classification", routes=["GET"])
def classification():
    return render_template("classification.html")

# ======================================================
# ROUTE FLASK: SUBMIT IMAGE FROM HTML FORM
# ======================================================
@app.route("/submit", methods=['POST'])
def get_output():
    if 'my_image' not in request.files:
        return render_template("classification.html", prediction="Tidak ada file diunggah", confidence=0)
        
    img = request.files['my_image']
    if img.filename == '':
        return render_template("classification.html", prediction="Tidak ada file terpilih", confidence=0)

    filename = secure_filename(img.filename)
    
    # Pastikan folder static ada
    if not os.path.exists("static"):
        os.makedirs("static")
        
    img_path = os.path.join("static", filename)
    img.save(img_path)

    # Jalankan Prediksi
    class_name, confidence = predict(img_path)

    if confidence < 0.70:
        prediction = "Bukan Motif Batik"
    else:
        prediction = class_name

    return render_template(
        "classification.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        img_path=img_path
    )

# ======================================================
# RUN (Menyesuaikan Port Dinamis Render)
# ======================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
