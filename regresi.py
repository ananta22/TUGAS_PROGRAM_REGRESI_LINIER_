import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Fungsi untuk model eksponensial
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Fungsi untuk menghitung RMS
def calculate_rms(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Membaca data dari spreadsheet
data = pd.read_csv('student_performance.csv')

# Ekstrak kolom yang diperlukan
TB = data['Hours Studied'].values
NT = data['Performance Index'].values

# Model Linear
linear_coeffs = np.polyfit(TB, NT, 1)
linear_model = np.poly1d(linear_coeffs)
NT_linear_pred = linear_model(TB)

# Model Eksponensial
exp_params, _ = curve_fit(exponential_model, TB, NT, p0=(1, 0.1))
NT_exponential_pred = exponential_model(TB, *exp_params)

# Menghitung galat RMS
rms_linear = calculate_rms(NT, NT_linear_pred)
rms_exponential = calculate_rms(NT, NT_exponential_pred)

# Menampilkan hasil galat RMS pada output
print(f"Nama: Rizky Ananta\nNIM:21120122120029\nGalat RMS Linear: {rms_linear:.4f}\nGalat RMS Eksponensial: {rms_exponential:.4f}")

# Plot data asli dan hasil regresi linear
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(TB, NT, s=10, label='Data Asli')  # Ukuran titik lebih kecil
plt.plot(TB, NT_linear_pred, color='red', label='Regresi Linear')
plt.xlabel('Durasi Waktu Belajar (jam)')
plt.ylabel('Nilai Ujian')
plt.title('Regresi Linear')
plt.yticks(np.arange(10, 105, 5))  # Indikator nilai ujian dari 10 hingga 100 dengan kelipatan 5
plt.legend()
plt.grid(True)
# Menampilkan galat RMS
plt.text(0.05, 0.95, f'RMS: {rms_linear:.4f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

# Plot data asli dan hasil regresi eksponensial
plt.subplot(1, 2, 2)
plt.scatter(TB, NT, s=10, label='Data Asli')  # Ukuran titik lebih kecil
plt.plot(TB, NT_exponential_pred, color='green', label='Regresi Eksponensial')
plt.xlabel('Durasi Waktu Belajar (jam)')
plt.ylabel('Nilai Ujian')
plt.title('Regresi Eksponensial')
plt.yticks(np.arange(10, 105, 5))  # Indikator nilai ujian dari 10 hingga 100 dengan kelipatan 5
plt.legend()
plt.grid(True)
# Menampilkan galat RMS
plt.text(0.05, 0.95, f'RMS: {rms_exponential:.4f}', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
