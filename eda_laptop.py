
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# ==========================================================
# 1. KONFIGURASI
# ==========================================================

FILE_DATA = "laptops_raw.csv"
OUTPUT_DIR = "eda_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================================
# 2. MEMBACA DATASET
# ==========================================================

print("=" * 65)
print("EXPLORATORY DATA ANALYSIS - LAPTOP PRICE DATASET")
print("=" * 65)

df = pd.read_csv(FILE_DATA)

print("\n[1] DATASET BERHASIL DIMUAT")
print(f"Jumlah baris  : {df.shape[0]:,}")
print(f"Jumlah kolom : {df.shape[1]}")

print("\nNama kolom:")
print(df.columns.tolist())

# ==========================================================
# 3. INFORMASI DATA
# ==========================================================

print("\n[2] INFORMASI DATA")
print("-" * 65)

df.info()

print("\n5 DATA PERTAMA:")
print(df.head())

print("\n5 DATA TERAKHIR:")
print(df.tail())

# ==========================================================
# 4. PEMERIKSAAN MISSING VALUES
# ==========================================================

print("\n[3] PEMERIKSAAN MISSING VALUES")
print("-" * 65)

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "Jumlah Missing": missing,
    "Persentase Missing": (missing / len(df)) * 100
})

print(missing_table)

# Grafik missing values
plt.figure(figsize=(12, 6))

missing_nonzero = missing[missing > 0].sort_values(ascending=False)

if not missing_nonzero.empty:
    sns.barplot(
        x=missing_nonzero.index,
        y=missing_nonzero.values
    )

    plt.title("Jumlah Missing Values per Kolom")
    plt.xlabel("Kolom")
    plt.ylabel("Jumlah Data Kosong")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "missing_values.png"),
        dpi=300
    )
    plt.show()
else:
    print("Tidak terdapat missing values.")

# ==========================================================
# 5. PEMERIKSAAN DATA DUPLIKAT
# ==========================================================

print("\n[4] PEMERIKSAAN DATA DUPLIKAT")
print("-" * 65)

duplicate_count = df.duplicated().sum()

print(f"Jumlah baris duplikat: {duplicate_count}")

# ==========================================================
# 6. KONVERSI HARGA KE NUMERIK
# ==========================================================

print("\n[5] PENGOLAHAN KOLOM HARGA")
print("-" * 65)

def convert_price(value):
    if pd.isna(value):
        return np.nan

    # Menghapus simbol mata uang, koma, dan karakter non-numerik
    cleaned = re.sub(r"[^\d.]", "", str(value))

    try:
        return float(cleaned)
    except ValueError:
        return np.nan

df["price_numeric"] = df["price"].apply(convert_price)

print("Contoh konversi harga:")
print(df[["price", "price_numeric"]].head(10))

# ==========================================================
# 7. STATISTIK DESKRIPTIF HARGA
# ==========================================================

print("\n[6] STATISTIK DESKRIPTIF HARGA LAPTOP")
print("-" * 65)

price_stats = df["price_numeric"].describe()

print(price_stats)

print(f"\nHarga terendah  : ₹{df['price_numeric'].min():,.0f}")
print(f"Harga tertinggi  : ₹{df['price_numeric'].max():,.0f}")
print("fHarga rata-rata  : ₹{df['price_numeric'].mean():,.2f}")
print("fHarga median     : ₹{df['price_numeric'].median():,.0f}")

# ==========================================================
# 8. DISTRIBUSI HARGA LAPTOP
# ==========================================================

print("\n[7] VISUALISASI DISTRIBUSI HARGA")

plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x="price_numeric",
    bins=30,
    kde=True
)

plt.title("Distribusi Harga Laptop")
plt.xlabel("Harga (INR)")
plt.ylabel("Frekuensi")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "distribusi_harga.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 9. BOXPLOT HARGA
# ==========================================================

print("\n[8] VISUALISASI BOXPLOT HARGA")

plt.figure(figsize=(12, 4))

sns.boxplot(
    x=df["price_numeric"]
)

plt.title("Boxplot Harga Laptop")
plt.xlabel("Harga (INR)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "boxplot_harga.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 10. EKSTRAKSI BRAND DARI NAMA MODEL
# ==========================================================

print("\n[9] ANALISIS BRAND LAPTOP")

def extract_brand(model):
    if pd.isna(model):
        return "Unknown"

    model = str(model).strip()

    if not model:
        return "Unknown"

    return model.split()[0].title()

df["brand"] = df["model"].apply(extract_brand)

brand_counts = df["brand"].value_counts()

print("\nJumlah data per brand:")
print(brand_counts)

# ==========================================================
# 11. TOP 10 BRAND LAPTOP
# ==========================================================

plt.figure(figsize=(12, 6))

top_brands = brand_counts.head(10)

sns.barplot(
    x=top_brands.values,
    y=top_brands.index
)

plt.title("10 Brand Laptop dengan Data Terbanyak")
plt.xlabel("Jumlah Laptop")
plt.ylabel("Brand")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_10_brand.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 12. RATA-RATA HARGA PER BRAND
# ==========================================================

print("\n[10] ANALISIS RATA-RATA HARGA PER BRAND")

brand_price = (
    df.groupby("brand")["price_numeric"]
    .mean()
    .sort_values(ascending=False)
)

print("\nRata-rata harga per brand:")
print(brand_price)

top_10_avg_price = brand_price.head(10)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top_10_avg_price.values,
    y=top_10_avg_price.index
)

plt.title("10 Brand dengan Rata-rata Harga Tertinggi")
plt.xlabel("Rata-rata Harga (INR)")
plt.ylabel("Brand")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "rata_rata_harga_brand.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 13. HUBUNGAN RATING DENGAN HARGA
# ==========================================================

print("\n[11] ANALISIS HUBUNGAN RATING DAN HARGA")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="rating",
    y="price_numeric",
    alpha=0.6
)

plt.title("Hubungan Rating dengan Harga Laptop")
plt.xlabel("Rating")
plt.ylabel("Harga (INR)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "rating_vs_harga.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 14. KORELASI RATING DAN HARGA
# ==========================================================

correlation = df[["rating", "price_numeric"]].corr()

print("\nKorelasi Rating dan Harga:")
print(correlation)

# ==========================================================
# 15. RINGKASAN HASIL EDA
# ==========================================================

print("\n" + "=" * 65)
print("RINGKASAN HASIL EDA")
print("=" * 65)

print(f"Total data laptop        : {len(df):,}")
print(f"Jumlah kolom             : {df.shape[1]}")
print(f"Data duplikat           : {duplicate_count}")
print(f"Total missing values     : {df.isnull().sum().sum():,}")
print(f"Harga minimum           : ₹{df['price_numeric'].min():,.0f}")
print(f"Harga maksimum           : ₹{df['price_numeric'].max():,.0f}")
print(f"Harga rata-rata          : ₹{df['price_numeric'].mean():,.2f}")
print(f"Jumlah brand            : {df['brand'].nunique()}")

print("\n10 Brand terbanyak:")
print(brand_counts.head(10))

print("\n10 Brand dengan harga rata-rata tertinggi:")
print(brand_price.head(10))

print("\nGrafik tersimpan di folder:", OUTPUT_DIR)

print("\nEDA SELESAI!")
print("=" * 65)

