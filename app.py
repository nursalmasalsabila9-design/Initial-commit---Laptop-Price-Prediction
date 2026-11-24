
import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Harga Laptop",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# KURS KONVERSI
# 1 INR = Rp196,12 (kurs referensi, dapat diperbarui)
# =========================================================
INR_TO_IDR = 196.12


def format_rupiah(value):
    """Format angka menjadi Rupiah."""
    return f"Rp{value:,.0f}".replace(",", ".")


def format_juta(value):
    """Format angka ke satuan juta Rupiah."""
    nilai_juta = value / 1_000_000
    return f"{nilai_juta:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# =========================================================
# CSS TAMPILAN
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E8EDF5;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 30px;
        background: linear-gradient(135deg, #17263B, #17202D);
        border: 1px solid #27384F;
        border-radius: 18px;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #F1F5F9;
        font-size: 32px;
        margin-bottom: 10px;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .hero p {
        color: #AAB8CA;
        font-size: 15px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .info-card {
        background-color: #171D27;
        border: 1px solid #293443;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
        height: auto;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    .info-label {
        color: #9AAFC5;
        font-size: 13px;
        margin-bottom: 10px;
        line-height: 1.5;
    }

    .info-value {
        color: #E8EDF5;
        font-size: 19px;
        font-weight: 700;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }

    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #E6EDF7;
        margin: 20px 0 15px 0;
    }

    .result-card {
        background: linear-gradient(135deg, #123B35, #142E2C);
        border: 1px solid #276C5D;
        border-radius: 18px;
        padding: 28px 20px;
        margin-top: 25px;
        margin-bottom: 15px;
        text-align: center;
    }

    .result-label {
        color: #A5CFC1;
        font-size: 15px;
        margin-bottom: 12px;
    }

    .result-price {
        color: #5EE0B0;
        font-size: 36px;
        font-weight: 700;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .result-million {
        color: #5EE0B0;
        font-size: 24px;
        font-weight: 600;
        margin-top: 8px;
    }

    .result-original {
        color: #C1D9D0;
        font-size: 14px;
        margin-top: 12px;
    }

    .result-description {
        color: #B3D7CC;
        font-size: 13px;
        margin-top: 12px;
        line-height: 1.7;
    }

    div[data-testid="stForm"] {
        background-color: #141A23;
        border: 1px solid #273342;
        padding: 20px;
        border-radius: 16px;
    }

    div[data-testid="stMetric"] {
        background-color: #171D27;
        padding: 15px;
        border: 1px solid #293443;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #AAB8CA;
    }

    div[data-testid="stMetricValue"] {
        color: #E8EDF5;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .stButton button {
        border-radius: 10px;
        min-height: 48px;
        font-weight: 600;
    }

    .note {
        color: #8290A3;
        font-size: 12px;
        line-height: 1.7;
    }

    hr {
        border-color: #273342;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOKASI FILE
# =========================================================
MODEL_PATH = os.path.join("models", "best_laptop_model.sav")
DATA_PATH = os.path.join("data_processed", "laptops_clean.csv")


# =========================================================
# MEMUAT MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================================================
# MEMUAT DATASET
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# =========================================================
# VALIDASI FILE
# =========================================================
if not os.path.exists(MODEL_PATH):
    st.error("File model tidak ditemukan: models/best_laptop_model.sav")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("File dataset tidak ditemukan: data_processed/laptops_clean.csv")
    st.stop()

try:
    model = load_model()
    df = load_data()

except Exception as e:
    st.error(f"Gagal memuat model atau dataset: {e}")
    st.stop()


# =========================================================
# FITUR MODEL
# =========================================================
categorical_features = [
    "processor",
    "cores",
    "ram",
    "storage",
    "display",
    "graphics",
    "os",
    "warranty"
]

numeric_features = ["rating"]

required_columns = categorical_features + numeric_features

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "Kolom berikut tidak ditemukan dalam dataset: "
        + ", ".join(missing_columns)
    )
    st.stop()


# =========================================================
# PILIHAN INPUT DINAMIS
# =========================================================
def get_options(column):
    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = sorted(
        [value for value in values.unique() if value],
        key=str.lower
    )

    if "Unknown" not in values:
        values.insert(0, "Unknown")

    return values


def format_option(value):
    if value == "Unknown":
        return "Tidak diketahui"
    return value


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>💻 Prediksi Harga Laptop</h1>
    <p>
        Perkirakan harga laptop berdasarkan spesifikasi perangkat.
        Sistem menggunakan model Machine Learning yang telah dilatih
        dengan data laptop untuk menghasilkan estimasi harga.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# INFORMASI APLIKASI - KARTU RESPONSIF
# =========================================================
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Jenis Proyek</div>
        <div class="info-value">Prediksi Harga Laptop</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Metode Analisis</div>
        <div class="info-value">Regresi Machine Learning</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Mata Uang Hasil</div>
        <div class="info-value">Rupiah Indonesia (IDR)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# =========================================================
# FORM SPESIFIKASI LAPTOP
# =========================================================
st.markdown(
    '<div class="section-title">Spesifikasi Laptop</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    st.markdown("#### Prosesor dan Performa")

    col1, col2 = st.columns(2)

    with col1:
        processor = st.selectbox(
            "Prosesor",
            get_options("processor"),
            format_func=format_option
        )

        cores = st.selectbox(
            "Jumlah Inti Prosesor",
            get_options("cores"),
            format_func=format_option
        )

    with col2:
        ram = st.selectbox(
            "Memori RAM",
            get_options("ram"),
            format_func=format_option
        )

        storage = st.selectbox(
            "Penyimpanan",
            get_options("storage"),
            format_func=format_option
        )

    st.markdown("#### Layar dan Grafis")

    col3, col4 = st.columns(2)

    with col3:
        display = st.selectbox(
            "Spesifikasi Layar",
            get_options("display"),
            format_func=format_option
        )

    with col4:
        graphics = st.selectbox(
            "Kartu Grafis",
            get_options("graphics"),
            format_func=format_option
        )

    st.markdown("#### Sistem dan Spesifikasi Tambahan")

    col5, col6 = st.columns(2)

    with col5:
        os_option = st.selectbox(
            "Sistem Operasi",
            get_options("os"),
            format_func=format_option
        )

    with col6:
        warranty = st.selectbox(
            "Garansi",
            get_options("warranty"),
            format_func=format_option
        )

    st.markdown("#### Penilaian Laptop")

    rating_min = float(df["rating"].min())
    rating_max = float(df["rating"].max())
    rating_default = float(df["rating"].median())

    rating = st.slider(
        "Rating Pengguna",
        min_value=rating_min,
        max_value=rating_max,
        value=rating_default,
        step=0.1
    )

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔍 Prediksi Harga Laptop",
        use_container_width=True
    )


# =========================================================
# PREDIKSI HARGA
# =========================================================
if submitted:

    input_data = pd.DataFrame([{
        "processor": processor,
        "cores": cores,
        "ram": ram,
        "storage": storage,
        "display": display,
        "graphics": graphics,
        "os": os_option,
        "warranty": warranty,
        "rating": rating
    }])

    try:
        prediction_inr = float(model.predict(input_data)[0])

        # Konversi INR ke IDR
        prediction_idr = prediction_inr * INR_TO_IDR

        # Konversi ke juta Rupiah
        prediction_million = prediction_idr / 1_000_000

        # Hasil utama
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Perkiraan Harga Laptop</div>
            <div class="result-price">{format_rupiah(prediction_idr)}</div>
            <div class="result-million">{format_juta(prediction_idr)} juta rupiah</div>
            <div class="result-original">
                Setara dengan sekitar ₹{prediction_inr:,.2f} Rupee India
            </div>
            <div class="result-description">
                Hasil estimasi berdasarkan spesifikasi laptop yang dipilih.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Ringkasan
        st.markdown("### Ringkasan Hasil Prediksi")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric(
                "Estimasi Harga (Rupiah)",
                format_rupiah(prediction_idr)
            )

        with summary_col2:
            st.metric(
                "Estimasi Harga (Juta)",
                f"{format_juta(prediction_idr)} juta"
            )

        with summary_col3:
            st.metric(
                "Rating Laptop",
                f"{rating:.1f} / 5"
            )

        # Tabel spesifikasi
        st.markdown("### Spesifikasi Laptop yang Dipilih")

        display_data = input_data.rename(columns={
            "processor": "Prosesor",
            "cores": "Jumlah Inti",
            "ram": "RAM",
            "storage": "Penyimpanan",
            "display": "Layar",
            "graphics": "Grafis",
            "os": "Sistem Operasi",
            "warranty": "Garansi",
            "rating": "Rating"
        })

        display_data = display_data.replace("Unknown", "Tidak diketahui")

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        # Informasi kurs
        st.markdown("### Informasi Konversi Mata Uang")

        st.info(
            f"Model menghasilkan harga dalam INR. "
            f"Hasil dikonversi menggunakan kurs acuan "
            f"1 INR = Rp{INR_TO_IDR:,.2f}. "
            f"Kurs dapat berubah sehingga hasil Rupiah merupakan perkiraan."
        )

        st.markdown("""
        <div class="note">
            Harga prediksi merupakan estimasi dari model Machine Learning,
            bukan harga jual resmi. Harga pasar dapat berbeda tergantung
            merek, kondisi perangkat, toko, dan perubahan nilai tukar.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediksi gagal dilakukan: {e}")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#8290A3; font-size:13px; line-height:1.8;">
    Prediksi Harga Laptop
    <br>
    Proyek Pembelajaran Mesin · Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)

