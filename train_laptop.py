
import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# 1. KONFIGURASI
# ==========================================================

DATA_FILE = "data_processed/laptops_clean.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "training_results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 70)
print("LAPTOP PRICE PREDICTION - MODEL TRAINING")
print("=" * 70)

# ==========================================================
# 2. MEMBACA DATASET
# ==========================================================

print("\n[1] MEMBACA DATASET")
print("-" * 70)

df = pd.read_csv(DATA_FILE)

print(f"Jumlah data     : {df.shape[0]}")
print(f"Jumlah kolom    : {df.shape[1]}")
print("\nKolom dataset:")
print(df.columns.tolist())

# ==========================================================
# 3. MENENTUKAN FITUR DAN TARGET
# ==========================================================

print("\n[2] MENENTUKAN FITUR DAN TARGET")
print("-" * 70)

# Target yang akan diprediksi
TARGET = "price_numeric"

# Fitur yang digunakan model
# Kolom model sengaja tidak digunakan untuk mengurangi overfitting
# karena nama model laptop memiliki banyak nilai unik.
# Brand juga tidak digunakan karena berasal dari ekstraksi nama model.

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

numeric_features = [
    "rating"
]

feature_columns = categorical_features + numeric_features

# Memastikan semua kolom tersedia
required_columns = feature_columns + [TARGET]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}"
    )

X = df[feature_columns].copy()
y = pd.to_numeric(df[TARGET], errors="coerce")

# Menghapus target yang tidak valid jika ada
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

print("\nFitur kategorikal:")
print(categorical_features)

print("\nFitur numerik:")
print(numeric_features)

print("\nTarget:")
print(TARGET)

print(f"\nJumlah data siap training: {len(X)}")

# ==========================================================
# 4. MEMBAGI DATA TRAINING DAN TESTING
# ==========================================================

print("\n[3] TRAIN TEST SPLIT")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Data training : {len(X_train)} ({100 - TEST_SIZE * 100:.0f}%)")
print(f"Data testing : {len(X_test)} ({TEST_SIZE * 100:.0f}%)")

# ==========================================================
# 5. PREPROCESSING PIPELINE
# ==========================================================

print("\n[4] MENYIAPKAN PREPROCESSING PIPELINE")
print("-" * 70)

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(transformers=[
    ("categorical", categorical_transformer, categorical_features),
    ("numeric", numeric_transformer, numeric_features)
])

print("Pipeline preprocessing berhasil disiapkan.")

# ==========================================================
# 6. MENENTUKAN MODEL
# ==========================================================

print("\n[5] DAFTAR MODEL MACHINE LEARNING")
print("-" * 70)

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
}

for model_name in models:
    print(f"- {model_name}")

# ==========================================================
# 7. TRAINING DAN EVALUASI MODEL
# ==========================================================

print("\n[6] PROSES TRAINING DAN EVALUASI")
print("-" * 70)

results = []
trained_pipelines = {}
prediction_results = {}

for model_name, model in models.items():

    print(f"\nTraining model: {model_name}")

    # Pipeline model + preprocessing
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Mengukur waktu training
    start_time = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Prediksi data testing
    predictions = pipeline.predict(X_test)

    # Evaluasi
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Training Time (s)": training_time
    })

    trained_pipelines[model_name] = pipeline
    prediction_results[model_name] = predictions

    print(f"MAE            : ₹{mae:,.2f}")
    print(f"RMSE           : ₹{rmse:,.2f}")
    print(f"R2 Score        : {r2:.4f}")
    print(f"Waktu Training  : {training_time:.2f} detik")

# ==========================================================
# 8. TABEL PERBANDINGAN MODEL
# ==========================================================

print("\n[7] HASIL PERBANDINGAN MODEL")
print("=" * 70)

results_df = pd.DataFrame(results)

# Urutkan berdasarkan MAE terendah
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

print(results_df.to_string(index=False))

# Simpan hasil evaluasi
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

# ==========================================================
# 9. MENENTUKAN MODEL TERBAIK
# ==========================================================

print("\n[8] MODEL TERBAIK")
print("-" * 70)

best_model_name = results_df.loc[0, "Model"]
best_pipeline = trained_pipelines[best_model_name]
best_predictions = prediction_results[best_model_name]

best_mae = results_df.loc[0, "MAE"]
best_rmse = results_df.loc[0, "RMSE"]
best_r2 = results_df.loc[0, "R2"]

print(f"Model terbaik : {best_model_name}")
print(f"MAE         : ₹{best_mae:,.2f}")
print(f"RMSE        : ₹{best_rmse:,.2f}")
print(f"R2 Score     : {best_r2:.4f}")

# ==========================================================
# 10. MENYIMPAN MODEL TERBAIK
# ==========================================================

print("\n[9] MENYIMPAN MODEL")
print("-" * 70)

model_path = os.path.join(MODEL_DIR, "best_laptop_model.sav")

joblib.dump(best_pipeline, model_path)

print(f"Model berhasil disimpan: {model_path}")

# ==========================================================
# 11. MENYIMPAN PREDIKSI DATA TESTING
# ==========================================================

print("\n[10] MENYIMPAN HASIL PREDIKSI")
print("-" * 70)

prediction_df = pd.DataFrame({
    "Harga Aktual": y_test.values,
    "Harga Prediksi": best_predictions,
    "Selisih": y_test.values - best_predictions,
    "Absolute Error": np.abs(y_test.values - best_predictions)
})

prediction_df.to_csv(
    os.path.join(OUTPUT_DIR, "best_model_predictions.csv"),
    index=False
)

print("\nContoh hasil prediksi:")
print(prediction_df.head(10).round(2).to_string(index=False))

# ==========================================================
# 12. GRAFIK PERBANDINGAN MAE
# ==========================================================

print("\n[11] GRAFIK MAE")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="MAE",
    y="Model"
)

plt.title("Perbandingan MAE Antar Model")
plt.xlabel("Mean Absolute Error (INR)")
plt.ylabel("Model")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "comparison_mae.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 13. GRAFIK PERBANDINGAN RMSE
# ==========================================================

print("\n[12] GRAFIK RMSE")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="RMSE",
    y="Model"
)

plt.title("Perbandingan RMSE Antar Model")
plt.xlabel("Root Mean Squared Error (INR)")
plt.ylabel("Model")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "comparison_rmse.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 14. GRAFIK R2 SCORE
# ==========================================================

print("\n[13] GRAFIK R2 SCORE")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="R2",
    y="Model"
)

plt.title("Perbandingan R2 Score Antar Model")
plt.xlabel("R2 Score")
plt.ylabel("Model")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "comparison_r2.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 15. GRAFIK HARGA AKTUAL VS PREDIKSI
# ==========================================================

print("\n[14] GRAFIK HARGA AKTUAL VS PREDIKSI")

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=y_test,
    y=best_predictions,
    alpha=0.65
)

# Garis prediksi ideal
min_price = min(y_test.min(), best_predictions.min())
max_price = max(y_test.max(), best_predictions.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.title(f"Aktual vs Prediksi - {best_model_name}")
plt.xlabel("Harga Aktual (INR)")
plt.ylabel("Harga Prediksi (INR)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 16. GRAFIK DISTRIBUSI ERROR
# ==========================================================

print("\n[15] GRAFIK DISTRIBUSI ERROR")

errors = y_test.values - best_predictions

plt.figure(figsize=(10, 6))

sns.histplot(
    errors,
    bins=30,
    kde=True
)

plt.axvline(0, linestyle="--")

plt.title(f"Distribusi Error Prediksi - {best_model_name}")
plt.xlabel("Error (Harga Aktual - Harga Prediksi)")
plt.ylabel("Frekuensi")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "error_distribution.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 17. RINGKASAN AKHIR
# ==========================================================

print("\n" + "=" * 70)
print("TRAINING SELESAI")
print("=" * 70)

print("\nRanking model berdasarkan MAE terendah:")
print(results_df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))

print("\nMODEL TERBAIK:")
print(f"Nama Model    : {best_model_name}")
print(f"MAE           : ₹{best_mae:,.2f}")
print(f"RMSE          : ₹{best_rmse:,.2f}")
print(f"R2 Score       : {best_r2:.4f}")

print("\nFile yang berhasil dibuat:")
print(f"- {model_path}")
print(f"- {OUTPUT_DIR}/model_comparison.csv")
print(f"- {OUTPUT_DIR}/best_model_predictions.csv")
print(f"- {OUTPUT_DIR}/comparison_mae.png")
print(f"- {OUTPUT_DIR}/comparison_rmse.png")
print(f"- {OUTPUT_DIR}/comparison_r2.png")
print(f"- {OUTPUT_DIR}/actual_vs_predicted.png")
print(f"- {OUTPUT_DIR}/error_distribution.png")

print("\nTRAINING DAN EVALUASI BERHASIL!")
print("=" * 70)

