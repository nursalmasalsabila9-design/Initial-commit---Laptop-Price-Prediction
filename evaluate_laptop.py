
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# 1. KONFIGURASI
# ==========================================================

DATA_FILE = "data_processed/laptops_clean.csv"
MODEL_FILE = "models/best_laptop_model.sav"
TRAINING_RESULTS_DIR = "training_results"
OUTPUT_DIR = "evaluation_results"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("EVALUASI MODEL LAPTOP PRICE PREDICTION")
print("=" * 65)

# ==========================================================
# 2. MEMBACA DATASET
# ==========================================================

print("\n[1] MEMBACA DATASET")
print("-" * 65)

df = pd.read_csv(DATA_FILE)

print(f"Jumlah dataset : {len(df)}")

# ==========================================================
# 3. MENENTUKAN FITUR DAN TARGET
# ==========================================================

print("\n[2] MENENTUKAN FITUR DAN TARGET")
print("-" * 65)

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
target_column = "price_numeric"

X = df[feature_columns].copy()
y = pd.to_numeric(df[target_column], errors="coerce")

valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# ==========================================================
# 4. MEMBAGI DATA TESTING
# ==========================================================

print("\n[3] MEMBAGI DATA TESTING")
print("-" * 65)

# Pembagian harus sama dengan Tahap 4 agar evaluasi
# menggunakan data testing yang sama dengan training.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Data training : {len(X_train)}")
print(f"Data testing : {len(X_test)}")

# ==========================================================
# 5. MEMUAT MODEL TERBAIK
# ==========================================================

print("\n[4] MEMUAT MODEL TERBAIK")
print("-" * 65)

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_FILE}. "
        "Jalankan train_laptop.py terlebih dahulu."
    )

best_pipeline = joblib.load(MODEL_FILE)

print("Model berhasil dimuat.")
print(f"Lokasi model : {MODEL_FILE}")

# ==========================================================
# 6. MELAKUKAN PREDIKSI
# ==========================================================

print("\n[5] MELAKUKAN PREDIKSI DATA TESTING")
print("-" * 65)

y_pred = best_pipeline.predict(X_test)

print("Prediksi berhasil dilakukan.")

# ==========================================================
# 7. MENGHITUNG METRIK EVALUASI
# ==========================================================

print("\n[6] METRIK EVALUASI")
print("-" * 65)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# MAPE dalam persen
mape = np.mean(
    np.abs((y_test.values - y_pred) / y_test.values)
) * 100

print(f"MAE  : ₹{mae:,.2f}")
print(f"MSE  : ₹{mse:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R2   : {r2:.4f}")
print(f"MAPE : {mape:.2f}%")

# ==========================================================
# 8. MENYIMPAN LAPORAN METRIK
# ==========================================================

print("\n[7] MENYIMPAN METRIK EVALUASI")
print("-" * 65)

metrics_df = pd.DataFrame([{
    "Model": type(best_pipeline.named_steps["model"]).__name__,
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "MAPE (%)": mape
}])

metrics_file = os.path.join(
    OUTPUT_DIR,
    "evaluation_metrics.csv"
)

metrics_df.to_csv(metrics_file, index=False)

print(metrics_df.to_string(index=False))

# ==========================================================
# 9. MENYIMPAN DETAIL HASIL PREDIKSI
# ==========================================================

print("\n[8] MENYIMPAN DETAIL PREDIKSI")
print("-" * 65)

prediction_df = pd.DataFrame({
    "Harga Aktual": y_test.values,
    "Harga Prediksi": y_pred,
    "Error": y_test.values - y_pred,
    "Absolute Error": np.abs(y_test.values - y_pred)
})

prediction_df["Error Persen"] = (
    prediction_df["Absolute Error"]
    / prediction_df["Harga Aktual"]
    * 100
)

prediction_file = os.path.join(
    OUTPUT_DIR,
    "evaluation_predictions.csv"
)

prediction_df.to_csv(prediction_file, index=False)

print("\n10 contoh prediksi:")
print(prediction_df.head(10).round(2).to_string(index=False))

# ==========================================================
# 10. GRAFIK AKTUAL VS PREDIKSI
# ==========================================================

print("\n[9] GRAFIK AKTUAL VS PREDIKSI")
print("-" * 65)

plt.figure(figsize=(9, 7))

sns.scatterplot(
    x=y_test,
    y=y_pred,
    alpha=0.65
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Prediksi ideal"
)

plt.title("Harga Aktual vs Harga Prediksi")
plt.xlabel("Harga Aktual (INR)")
plt.ylabel("Harga Prediksi (INR)")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 11. GRAFIK DISTRIBUSI ERROR
# ==========================================================

print("\n[10] GRAFIK DISTRIBUSI ERROR")
print("-" * 65)

errors = y_test.values - y_pred

plt.figure(figsize=(10, 6))

sns.histplot(
    errors,
    bins=30,
    kde=True
)

plt.axvline(
    0,
    linestyle="--",
    label="Error = 0"
)

plt.title("Distribusi Error Model")
plt.xlabel("Error (Harga Aktual - Harga Prediksi)")
plt.ylabel("Frekuensi")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "error_distribution.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 12. GRAFIK ABSOLUTE ERROR
# ==========================================================

print("\n[11] GRAFIK ABSOLUTE ERROR")
print("-" * 65)

plt.figure(figsize=(10, 6))

sns.histplot(
    prediction_df["Absolute Error"],
    bins=30,
    kde=True
)

plt.title("Distribusi Absolute Error")
plt.xlabel("Absolute Error (INR)")
plt.ylabel("Frekuensi")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "absolute_error_distribution.png"),
    dpi=300
)

plt.show()

# ==========================================================
# 13. FEATURE IMPORTANCE
# ==========================================================

print("\n[12] ANALISIS FEATURE IMPORTANCE")
print("-" * 65)

trained_model = best_pipeline.named_steps["model"]
trained_preprocessor = best_pipeline.named_steps["preprocessor"]

try:
    feature_names = trained_preprocessor.get_feature_names_out()

    # Model berbasis tree
    if hasattr(trained_model, "feature_importances_"):

        importance_values = trained_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance_values
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(20)

        importance_title = "Feature Importance"

    # Model Linear Regression
    elif hasattr(trained_model, "coef_"):

        importance_values = np.ravel(trained_model.coef_)

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(importance_values)
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(20)

        importance_title = "Top 20 Feature berdasarkan Absolut Koefisien"

    else:
        importance_df = pd.DataFrame()
        importance_title = None

    if not importance_df.empty:

        importance_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "feature_importance.csv"
            ),
            index=False
        )

        print("\n20 fitur teratas:")
        print(importance_df.to_string(index=False))

        plt.figure(figsize=(12, 8))

        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature"
        )

        plt.title(importance_title)
        plt.xlabel("Nilai Importance")
        plt.ylabel("Fitur")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "feature_importance.png"
            ),
            dpi=300
        )

        plt.show()

    else:
        print("Feature importance tidak tersedia untuk model ini.")

except Exception as error:
    print(f"Feature importance tidak dapat dihitung: {error}")

# ==========================================================
# 14. ANALISIS KECENDERUNGAN ERROR
# ==========================================================

print("\n[13] ANALISIS KECENDERUNGAN ERROR")
print("-" * 65)

mean_error = np.mean(errors)
median_absolute_error = np.median(np.abs(errors))
max_error = np.max(np.abs(errors))

print(f"Rata-rata error       : ₹{mean_error:,.2f}")
print(f"Median absolute error : ₹{median_absolute_error:,.2f}")
print(f"Error terbesar        : ₹{max_error:,.2f}")

if mean_error > 0:
    print("Model cenderung UNDERPREDICT (prediksi lebih rendah).")
elif mean_error < 0:
    print("Model cenderung OVERPREDICT (prediksi lebih tinggi).")
else:
    print("Rata-rata prediksi sama dengan harga aktual.")

# ==========================================================
# 15. RINGKASAN EVALUASI
# ==========================================================

print("\n" + "=" * 65)
print("EVALUASI MODEL SELESAI")
print("=" * 65)

print(f"Jumlah data testing : {len(y_test)}")
print(f"MAE                : ₹{mae:,.2f}")
print(f"RMSE               : ₹{rmse:,.2f}")
print("fR2 Score            : {r2:.4f}")
print(f"MAPE               : {mape:.2f}%")

print("\nFile hasil evaluasi:")
print(f"- {metrics_file}")
print(f"- {prediction_file}")
print(f"- {OUTPUT_DIR}/actual_vs_predicted.png")
print(f"- {OUTPUT_DIR}/error_distribution.png")
print(f"- {OUTPUT_DIR}/absolute_error_distribution.png")
print(f"- {OUTPUT_DIR}/feature_importance.csv (jika tersedia)")
print(f"- {OUTPUT_DIR}/feature_importance.png (jika tersedia)")

print("\nSemua proses evaluasi selesai.")
print("=" * 65)

