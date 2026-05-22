from flask import Flask, render_template, request
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

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
# PREDIKSI
# ======================================================

def predict_label(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    confidence = np.max(pred)

    pred_class = np.argmax(pred, axis=1)

    # CEK NON BATIK
    if confidence < 0.70:
        return "Bukan Motif Batik", confidence

    return dic[pred_class[0]], confidence

# ======================================================
# HALAMAN DATA BATIK
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
            "filosofi": "Melambangkan karakter masyarakat yang terbuka dalam menerima perbedaan namun tetap menjaga batas privasi. Bentuknya yang menyempit ke atas melambangkan kerendahan hati di hadapan Sang Pencipta. Simbol keramahan yang berlandaskan pada keteguhan prinsip adat dan agama."
        },

        {
            "nama": "Bali - Barong",
            "pulau" : "bali_ntt_ntb",
            "gambar": [
                "/static/batik/Bali Barong/1.jpg",
                "/static/batik/Bali Barong/2.jpg",
                "/static/batik/Bali Barong/3.jpg"
            ],
            "filosofi": "Merepresentasikan filosofi Rwa Bhineda, yakni keseimbangan abadi antara kebaikan dan kejahatan di dunia. Sosok Barong sebagai pelindung spiritual masyarakat dari segala bentuk gangguan negatif. Lambang kemenangan dharma (kebajikan) yang memberikan rasa aman serta kedamaian hidup."
        },

        {
            "nama": "Bali - Merak",
            "pulau" : "bali_ntt_ntb",
            "gambar": [
                "/static/batik/Bali Merak/1.jpg",
                "/static/batik/Bali Merak/2.jpg",
                "/static/batik/Bali Merak/3.jpg"
            ],
            "filosofi": "Simbol keindahan estetika yang megah sekaligus melambangkan kesucian dan martabat yang tinggi. Gerakan molek burung merak mencerminkan keasrian alam Pulau Dewata yang penuh dengan harmoni. Menjadi doa bagi pemakainya agar memancarkan aura kepemimpinan yang anggun dan berwibawa."
        },

        {
            "nama": "DKI Jakarta - Ondel-Ondel",
            "pulau" : "Jawa",
            "gambar": [
                "/static/batik/DKI Ondel-Ondel/1.jpg",
                "/static/batik/DKI Ondel-Ondel/2.jpg",
                "/static/batik/DKI Ondel-Ondel/3.jpg"
            ],
            "filosofi": "MRepresentasi dari kekuatan penolak bala atau penjaga dari segala marabahaya dan kesialan dalam hidup. Menonjolkan semangat kegembiraan dan keceriaan masyarakat Betawi yang terbuka serta humoris. Simbol perlindungan sosial agar warga senantiasa hidup dalam ketenangan dan kebersamaan."
        },

        {
            "nama": "Jawa Barat - Megamendung",
            "pulau" : "Jawa",
            "gambar": [
                "/static/batik/Jawa Barat Megamendung/1.jpg",
                "/static/batik/Jawa Barat Megamendung/2.jpg",
                "/static/batik/Jawa Barat Megamendung/3.jpg"
            ],
            "filosofi": "Pola awan yang bergradasi melambangkan dunia atas yang luas serta pentingnya menahan amarah dalam kondisi apa pun. Filosofi ini mengajarkan kesabaran, kebijaksanaan, dan pikiran yang sejuk seperti suasana saat mendung. Pemakainya diharapkan menjadi sosok yang tenang dan mampu mengayomi sekelilingnya."
        },

        {
            "nama": "Jawa Timur - Pring",
            "pulau" : "Jawa",
            "gambar": [
                "/static/batik/Jawa Timur Pring/1.jpg",
                "/static/batik/Jawa Timur Pring/2.jpg",
                "/static/batik/Jawa Timur Pring/3.jpg"
            ],
            "filosofi": "Kata pring (bambu) melambangkan kekuatan yang fleksibel dan kemampuan beradaptasi di tengah badai kehidupan. Rumpun bambu yang hidup bersama menyimbolkan kerukunan dan kekuatan dalam ikatan kemasyarakatan. Menjadi pengingat untuk tetap teguh berdiri namun tetap rendah hati meski telah mencapai puncak."
        },

        {
            "nama": "Kalimantan - Dayak",
            "pulau" : "Kalimantan",
            "gambar": [
                "/static/batik/Kalimantan Dayak/1.jpg",
                "/static/batik/Kalimantan Dayak/2.jpg",
                "/static/batik/Kalimantan Dayak/3.jpg"
            ],
            "filosofi": "Menggambarkan hubungan sakral yang tak terputus antara manusia, alam liar, dan roh-roh para leluhur. Motifnya yang tegas mencerminkan keberanian, perlindungan, dan identitas kesukuan yang sangat kuat. Melambangkan harmoni hutan sebagai sumber kehidupan yang harus dijaga dengan kehormatan."
        },

        {
            "nama": "Lampung - Gajah",
            "pulau" : "Sumatera",
            "gambar": [
                "/static/batik/Lampung Gajah/1.jpg",
                "/static/batik/Lampung Gajah/2.jpg",
                "/static/batik/Lampung Gajah/3.jpg"
            ],
            "filosofi": "Gajah sebagai simbol kecerdasan, kekuatan fisik yang besar, serta kesetiaan kepada kelompoknya. Dalam budaya lokal, gajah dipandang sebagai hewan agung yang melambangkan kebesaran jiwa seorang pemimpin. Menjadi harapan agar pemakainya memiliki martabat tinggi dan mampu mengemban tanggung jawab besar."
        },

        {
            "nama": "Madura - Mataketeran",
            "pulau" : "Jawa",
            "gambar": [
                "/static/batik/Madura Mataketeran/1.jpg",
                "/static/batik/Madura Mataketeran/2.jpg",
                "/static/batik/Madura Mataketeran/3.jpg"
            ],
            "filosofi": "Cerminan karakter masyarakat Madura yang ekspresif, jujur, apa adanya, dan memiliki semangat kerja yang tinggi. Warna-warnanya yang berani melambangkan ketegasan dalam bersikap dan keberanian menghadapi tantangan. Motif ini menyimbolkan gairah hidup yang dinamis dan pantang menyerah dalam segala situasi."
        },

        {
            "nama": "Maluku - Pala",
            "pulau" : "Maluku",
            "gambar": [
                "/static/batik/Maluku Pala/1.jpg",
                "/static/batik/Maluku Pala/2.jpg",
                "/static/batik/Maluku Pala/3.jpg"
            ],
            "filosofi": "Rempah pala melambangkan kekayaan alam nusantara yang pernah menjadi pusat perhatian dunia dan simbol kemakmuran. Mencerminkan sejarah panjang perjuangan bangsa serta ketahanan ekonomi masyarakat kepulauan. Filosofi ini membawa pesan tentang rasa syukur atas berkah bumi yang memberikan kehidupan."
        },

        {
            "nama": "NTB - Lumbung",
            "pulau" : "bali_ntt_ntb",
            "gambar": [
                "/static/batik/NTB Lumbung/1.jpg",
                "/static/batik/NTB Lumbung/2.jpg",
                "/static/batik/NTB Lumbung/3.jpg"
            ],
            "filosofi": "Mengambil bentuk bangunan penyimpanan padi yang menyimbolkan kesejahteraan dan ketahanan pangan masyarakat. Mengajarkan nilai tentang pentingnya menabung, berhemat, serta kesiapan dalam menghadapi masa depan. Lambang kemandirian ekonomi dan rasa syukur atas hasil panen yang melimpah."
        },

        {
            "nama": "Papua - Asmat",
            "pulau" : "Papua",
            "gambar": [
                "/static/batik/Papua Asmat/1.jpg",
                "/static/batik/Papua Asmat/2.jpg",
                "/static/batik/Papua Asmat/3.jpg"
            ],
            "filosofi": "Ukiran khas ini melambangkan penghormatan terdalam kepada arwah nenek moyang yang melindungi suku mereka. Setiap garisnya mengandung kekuatan magis dan semangat maskulinitas sebagai pelindung keluarga. Menjadi simbol identitas jati diri yang terhubung erat dengan tanah kelahiran dan tradisi."
        },


        {
            "nama": "Papua - Cendrawasih",
            "pulau" : "Papua",
            "gambar": [
                "/static/batik/Papua Cendrawasih/1.jpg",
                "/static/batik/Papua Cendrawasih/2.jpg",
                "/static/batik/Papua Cendrawasih/3.jpg"
            ],
            "filosofi": "Burung surga ini adalah simbol kecantikan yang sakral, kemuliaan, serta keajaiban alam Papua yang tiada tara. Melambangkan derajat yang tinggi dan harapan untuk selalu membawa kebaikan bagi lingkungan sekitar. Pemakainya diharapkan memiliki keelokan budi pekerti yang membuat orang lain merasa teduh."
        },

        {
            "nama": "Papua - Tifa",
            "pulau" : "Papua",
            "gambar": [
                "/static/batik/Papua Tifa/1.jpg",
                "/static/batik/Papua Tifa/2.jpg",
                "/static/batik/Papua Tifa/3.jpg"
            ],
            "filosofi": "Alat musik Tifa melambangkan panggilan jiwa untuk bersatu dan harmoni yang tercipta lewat nada-nada kehidupan. Mencerminkan semangat gotong royong dan kebersamaan dalam merayakan setiap peristiwa penting di suku. Simbol komunikasi yang jujur serta pengikat tali persaudaraan antarwarga yang sangat kuat."
        },

        {
            "nama": "Solo - Parang",
            "pulau" : "Jawa",
            "gambar": [
                "/static/batik/Solo Parang/1.jpg",
                "/static/batik/Solo Parang/2.jpg",
                "/static/batik/Solo Parang/3.jpg"
            ],
            "filosofi": "Motif lereng yang menyerupai ombak laut ini melambangkan semangat yang tak pernah padam dan pantang menyerah. Garis diagonalnya menyimbolkan jalinan yang berkesinambungan antara usaha manusia dan restu Tuhan. Sebagai simbol kekuasaan dan ketangkasan bagi mereka yang memiliki jiwa pemimpin sejati."
        },

        {
            "nama": "Sulawesi Selatan - Lontara",
            "pulau" : "Sulawesi",
            "gambar": [
                "/static/batik/Sulawesi Selatan Lontara/1.jpg",
                "/static/batik/Sulawesi Selatan Lontara/2.jpg",
                "/static/batik/Sulawesi Selatan Lontara/3.jpg"
            ],
            "filosofi": "Penggunaan aksara kuno ini melambangkan penghormatan tinggi terhadap sejarah, literasi, dan ilmu pengetahuan. Mencerminkan martabat kaum terpelajar yang menjunjung tinggi etika dan nilai-nilai leluhur Bugis-Makassar. Simbol kearifan lokal dalam bertutur kata serta bertindak secara bijaksana."
        },

        {
            "nama": "Sumatera Barat - Rumah Minang",
            "pulau" : "Sumatera",
            "gambar": [
                "/static/batik/Sumatera Barat Rumah Minang/1.jpg",
                "/static/batik/Sumatera Barat Rumah Minang/2.jpg",
                "/static/batik/Sumatera Barat Rumah Minang/3.jpg"
            ],
            "filosofi": "Melambangkan prinsip musyawarah dan mufakat yang dijunjung tinggi dalam struktur masyarakat adat. Bentuk atap gonjong menyimbolkan hubungan antara manusia dengan alam serta Sang Pencipta di langit. Filosofi ini menekankan pentingnya peran keluarga dan keadilan dalam kehidupan sosial."
        },

        {
            "nama": "Sumatera Utara - Boraspati",
            "pulau" : "Sumatera",
            "gambar": [
                "/static/batik/Sumatera Utara Boraspati/1.jpg",
                "/static/batik/Sumatera Utara Boraspati/2.jpg",
                "/static/batik/Sumatera Utara Boraspati/3.jpg"
            ],
            "filosofi": "Sosok cicak pelindung yang melambangkan kemakmuran, kesuburan, serta penjagaan terhadap harta benda rumah tangga. Dalam budaya Batak, ini adalah simbol kebijakan dalam mencari nafkah dan menjaga keutuhan keluarga. Doa agar pemiliknya selalu dilimpahi keberuntungan dan terlindung dari niat buruk."
        },

        {
            "nama": "Yogyakarta - Kawung",
            "pulau" : "Jawa",
            "gambar": [
                "/static/batik/Yogyakarta Kawung/1.jpg",
                "/static/batik/Yogyakarta Kawung/2.jpg",
                "/static/batik/Yogyakarta Kawung/3.jpg"
            ],
            "filosofi": "Bentuk empat lingkaran yang berpusat melambangkan kesucian hati dan kejujuran yang harus tetap bersih. Terinspirasi dari buah kolang-kaling yang putih di dalam, mengajarkan agar manusia bermanfaat bagi sesama. Simbol keseimbangan mikrokosmos dan makrokosmos serta pengendalian hawa nafsu."
        },

        {
            "nama": "Yogyakarta - Parang",
            "pulau" : "Jawa",
            "gambar": [
                "/static/batik/Yogyakarta Parang/1.jpg",
                "/static/batik/Yogyakarta Parang/2.jpg",
                "/static/batik/Yogyakarta Parang/3.jpg"
            ],
            "filosofi": "Berbeda sedikit dengan versi Solo, Parang Yogya menekankan pada ketegasan garis dan otoritas kedaulatan. Melambangkan ombak samudera sebagai kekuatan alam yang mahabesar sekaligus ujian hidup yang terus datang. Mengajarkan bahwa manusia harus terus bergerak maju tanpa henti demi mencapai kemuliaan."
        }

    ]

    return render_template(
        "data_batik.html",
        data_batik=data_batik
    )

# ======================================================
# HALAMAN CLASSIFICATION
# ======================================================

@app.route("/classification")
def classification():

    return render_template("classification.html")

# ======================================================
# SUBMIT IMAGE
# ======================================================

@app.route("/submit", methods=['POST'])
def get_output():

    # AMBIL FILE
    img = request.files['my_image']

    # AMANKAN NAMA FILE
    filename = secure_filename(img.filename)

    # SIMPAN KE FOLDER STATIC
    img_path = os.path.join("static", filename)

    img.save(img_path)

    # PREDIKSI
    prediction, confidence = predict_label(img_path)

    return render_template(
        "classification.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        img_path=img_path
    )

# ======================================================
# RUN
# ======================================================

if __name__ == '__main__':
    app.run(debug=True)