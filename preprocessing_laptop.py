
import pandas as pd
import numpy as np
import re
import os

# ==========================================================
# 1. KONFIGURASI
# ==========================================================

FILE_DATA = "laptops_raw.csv"
OUTPUT_DIR = "data_processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "laptops_clean.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("DATA PREPROCESSING - LAPTOP PRICE PREDICTION")
print("=" * 65)

# ==========================================================
# 2. MEMBACA DATA
# ==========================================================

df = pd.read_csv(FILE_DATA)

print("\n[1] DATA AWAL")
print("-" * 65)

print(f"Jumlah data awal : {len(df)}")
print(f"Jumlah kolom     : {df.shape[1]}")

print("\nNama kolom:")
print(df.columns.tolist())

# ==========================================================
# 3. MEMBERSIHKAN NAMA KOLOM
# ==========================================================

df.columns = df.columns.str.strip().str.lower()

# ==========================================================
# 4. MENGHAPUS DATA DUPLIKAT
# ==========================================================

print("\n[2] PENGHAPUS DATA DUPLIKAT")
print("-" * 65)

duplicate_count = df.duplicated().sum()

print(f"Jumlah duplikat ditemukan : {duplicate_count}")

df = df.drop_duplicates().copy()

print(f"Jumlah data setelah penghapusan : {len(df)}")

# ==========================================================
# 5. MEMBERSIHKAN KOLOM HARGA
# ==========================================================

print("\n[3] KONVERSI HARGA KE NUMERIK")
print("-" * 65)

def convert_price(value):
    if pd.isna(value):
        return np.nan

    cleaned = re.sub(r"[^\d.]", "", str(value))

    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan

df["price_numeric"] = df["price"].apply(convert_price)

invalid_price = df["price_numeric"].isna().sum()

print(f"Harga tidak valid : {invalid_price}")

# Data tanpa target harga tidak dapat digunakan untuk supervised learning
df = df.dropna(subset=["price_numeric"]).copy()

print(f"Jumlah data setelah pemeriksaan harga : {len(df)}")

# ==========================================================
# 6. MEMBERSIHKAN RATING
# ==========================================================

print("\n[4] PEMBERSIHAN RATING")
print("-" * 65)

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

missing_rating = df["rating"].isna().sum()

print(f"Rating kosong : {missing_rating}")

if df["rating"].notna().any():
    median_rating = df["rating"].median()
    df["rating"] = df["rating"].fillna(median_rating)
else:
    # Jika seluruh rating kosong, gunakan nilai default
    df["rating"] = df["rating"].fillna(0)

print("Rating berhasil ditangani.")

# ==========================================================
# 7. MEMBERSIHKAN DATA KATEGORIK
# ==========================================================

print("\n[5] PEMBERSIHAN DATA SPESIFIKASI")
print("-" * 65)

categorical_columns = [
    "model",
    "processor",
    "cores",
    "ram",
    "storage",
    "display",
    "graphics",
    "os",
    "warranty"
]

for column in categorical_columns:

    if column in df.columns:

        # Mengubah data menjadi string
        df[column] = df[column].astype("string")

        # Menghapus spasi awal dan akhir
        df[column] = df[column].str.strip()

        # Menangani data kosong dan teks kosong
        df[column] = df[column].replace(
            ["", "nan", "None", "<NA>"],
            pd.NA
        )

        # Mengisi data kosong
        df[column] = df[column].fillna("Unknown")

print("Data spesifikasi berhasil dibersihkan.")

# ==========================================================
# 8. EKSTRAKSI BRAND
# ==========================================================

print("\n[6] EKSTRAKSI BRAND")
print("-" * 65)

def extract_brand(model):
    if pd.isna(model) or str(model).strip() == "":
        return "Unknown"

    return str(model).strip().split()[0].title()

df["brand"] = df["model"].apply(extract_brand)

print("Brand berhasil diekstrak.")
print("\nContoh hasil:")
print(df[["model", "brand"]].head(10))

# ==========================================================
# 9. MEMERIKSA HASIL PREPROCESSING
# ==========================================================

print("\n[7] PEMERIKSAAN DATA AKHIR")
print("-" * 65)

print("Jumlah data akhir :", len(df))
print("Jumlah kolom     :", df.shape[1])

print("\nMissing values:")
print(df.isnull().sum())

print("\nData akhir:")
print(df.head())

# ==========================================================
# 10. MENYIMPAN DATA CLEAN
# ==========================================================

print("\n[8] MENYIMPAN DATA")
print("-" * 65)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Data berhasil disimpan: {OUTPUT_FILE}")

# ==========================================================
# 11. RINGKASAN PREPROCESSING
# ==========================================================

print("\n" + "=" * 65)
print("PREPROCESSING SELESAI")
print("=" * 65)

print(f"Data awal       : {len(pd.read_csv(FILE_DATA))}")
print(f"Data akhir     : {len(df)}")
print(f"Data terhapus  : {len(pd.read_csv(FILE_DATA)) - len(df)}")
print(f"Jumlah fitur    : {df.shape[1] - 1}")
print(f"Target         : price_numeric")
print(f"Harga minimum  : ₹{df['price_numeric'].min():,.0f}")
print(f"Harga maksimum  : ₹{df['price_numeric'].max():,.0f}")
print(f"Harga rata-rata : ₹{df['price_numeric'].mean():,.2f}")

print("\nFile output:")
print(OUTPUT_FILE)

print("\nData siap digunakan untuk tahap Machine Learning.")
print("=" * 65)

