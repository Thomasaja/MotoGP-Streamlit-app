# -*- coding: utf-8 -*-
"""Analisis dan Prediksi Performa MotoGP Riders Menggunakan Algoritma LSTM.ipynb

Adapted for Streamlit from Google Colab Notebook.
"""

# Import Library
import numpy as np
import pandas as pd
import math
import re 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

import streamlit as st
import joblib 
import io

# --- KONFIGURASI APLIKASI STREAMLIT ---
st.set_page_config(
    page_title="Prediksi Performa MotoGP Riders",
    page_icon="🏍️",
    layout="wide" # Menggunakan layout lebar agar lebih banyak ruang
)

st.title("🏍️ Prediksi Performa MotoGP Riders")
st.markdown("Aplikasi ini menganalisis data balapan MotoGP dan memprediksi posisi finish menggunakan model LSTM.")

# --- PENTING: BAGIAN MEMUAT DATASET DAN MODEL/SCALER ---
# Di Streamlit, Anda biasanya memuat data dari file lokal
# atau memungkinkan pengguna mengunggahnya.

# 1. Pemuatan Dataset
# Jika Anda memiliki file CSV lokal, ganti st.file_uploader dengan pd.read_csv langsung.
# Contoh: df = pd.read_csv("nama_file_dataset_anda.csv")

uploaded_file = st.file_uploader("Pilih file CSV dataset MotoGP Anda", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data berhasil dimuat!")
    st.subheader("Preview Data:")
    st.dataframe(df.head())

    # Lanjutkan pemrosesan data hanya jika file sudah diunggah
    st.subheader("Ringkasan Data:")
    st.write("Jumlah baris, kolom:", df.shape)
    st.write("Info data:")
    st.text(df.info(buf=io.StringIO())) # Mengarahkan info() ke st.text
    st.write("Statistik Deskriptif:")
    st.dataframe(df.describe())

    st.write("Distribusi Kolom 'position':")
    st.write(df['position'].value_counts().sort_index())
    st.write("Distribusi Kolom 'points':")
    st.write(df['points'].value_counts().sort_index())

    # --- PENTING: Pra-pemrosesan Data (hanya jika data berhasil dimuat) ---
    # Ini harus dilakukan setiap kali data baru diunggah
    df = df[df['position'] > 0].copy() # Tambahkan .copy() untuk menghindari SettingWithCopyWarning
    df = df[['year', 'sequence', 'position', 'points', 'rider_name']].copy() # Tambahkan .copy()
    df = df.sort_values(by=['year', 'sequence'])

    # --- PENTING: Pemuatan Model dan Scaler ---
    # Model dan scaler harus dilatih dan disimpan SEBELUM deployment ke Streamlit.
    # Misalnya, Anda latih di Colab, lalu simpan:
    # model.save('my_lstm_model.h5')
    # joblib.dump(scaler, 'my_minmax_scaler.pkl')

    # Kemudian, letakkan file 'my_lstm_model.h5' dan 'my_minmax_scaler.pkl' di direktori yang sama
    # dengan file Streamlit Anda, atau di sub-folder yang dapat diakses.

    try:
        # Muat scaler
        scaler_path = 'my_minmax_scaler.pkl' # Sesuaikan nama file scaler Anda
        scaler = joblib.load(scaler_path)
        st.sidebar.success(f"Scaler berhasil dimuat dari '{scaler_path}'")

        # Muat model
        model_path = 'my_lstm_model.h5' # Sesuaikan nama file model Anda
        model = load_model(model_path)
        st.sidebar.success(f"Model LSTM berhasil dimuat dari '{model_path}'")

        # Lanjutkan dengan preprocessing data untuk model
        # Normalisasi
        scaled_data = scaler.fit_transform(df[['position', 'points']])

        # Fungsi membuat dataset time series
        def create_dataset(dataset, time_step=5):
            X, y = [], []
            for i in range(len(dataset) - time_step):
                X.append(dataset[i:i+time_step])
                y.append(dataset[i+time_step][0])   # prediksi posisi
            return np.array(X), np.array(y)

        time_step = 5 # Definisi time_step untuk prediksi
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 2)   # 2 fitur: position, points

        # Split data (ini hanya untuk evaluasi, model sudah dilatih)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Prediksi (ini hanya untuk ditampilkan sebagai metrik)
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)

        # Inverse scaling hanya untuk posisi
        train_position = scaler.inverse_transform(np.hstack((train_predict, np.zeros_like(train_predict))))
        test_position = scaler.inverse_transform(np.hstack((test_predict, np.zeros_like(test_predict))))
        y_train_inv = scaler.inverse_transform(np.hstack((y_train.reshape(-1,1), np.zeros_like(y_train.reshape(-1,1)))))
        y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros_like(y_test.reshape(-1,1)))))

        # Hitung RMSE
        train_rmse = math.sqrt(mean_squared_error(y_train_inv[:,0], train_position[:,0]))
        test_rmse = math.sqrt(mean_squared_error(y_test_inv[:,0], test_position[:,0]))

        st.subheader("Metrik Model (Evaluasi Data Tersedia):")
        st.write(f"Train RMSE: {train_rmse:.2f}")
        st.write(f"Test RMSE : {test_rmse:.2f}")

        # --- PENTING: Plot Prediksi vs Aktual (Menggunakan st.pyplot) ---
        fig_pred_actual = plt.figure(figsize=(12,6))
        plt.plot(y_test_inv[:100, 0], label='Actual Position')
        plt.plot(test_position[:100, 0], label='Predicted Position')
        plt.title('Prediksi vs Aktual Posisi Pembalap (Contoh 100 Data)')
        plt.xlabel('Data Ke-')
        plt.ylabel('Posisi')
        plt.legend()
        plt.grid(True)
        st.pyplot(fig_pred_actual) # Tampilkan plot di Streamlit

    except FileNotFoundError:
        st.error("Error: File model (my_lstm_model.h5) atau scaler (my_minmax_scaler.pkl) tidak ditemukan.")
        st.warning("Pastikan Anda sudah melatih model dan scaler di Colab, lalu mengunduh dan meletakkannya di direktori yang sama dengan aplikasi Streamlit ini.")
        st.stop() # Hentikan eksekusi jika model/scaler tidak ada
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau scaler: {e}")
        st.stop()


    # --- PENTING: Konversi IPYWidgets ke Streamlit Widgets ---

    # Menggunakan st.sidebar untuk dropdown seleksi
    st.sidebar.subheader("Pilih Pembalap & Race")

    # Dropdown Pembalap
    rider_options = sorted(df['rider_name'].dropna().unique())
    selected_rider = st.sidebar.selectbox(
        'Pembalap:',
        options=rider_options
    )

    # Filter tahun berdasarkan pembalap yang dipilih
    years_for_rider = sorted(df[df['rider_name'] == selected_rider]['year'].unique())
    selected_year = st.sidebar.selectbox(
        'Tahun:',
        options=years_for_rider
    )

    # Filter race berdasarkan pembalap dan tahun yang dipilih
    races_for_selection = sorted(df[(df['rider_name'] == selected_rider) & (df['year'] == selected_year)]['sequence'].unique())
    selected_race = st.sidebar.selectbox(
        'Race Ke-:',
        options=races_for_selection
    )

    # --- PENTING: Logika Dashboard (akan dieksekusi ulang saat seleksi berubah) ---
    st.subheader(f"🏁 Dashboard MotoGP: {selected_rider} ({selected_year}) - Hingga Race ke-{selected_race}")

    # Check if all required selections are made
    if selected_rider and selected_year is not None and selected_race is not None:

        rider_df = df[
            (df['rider_name'] == selected_rider) &
            (df['year'] == selected_year) &
            (df['position'] > 0)
        ].copy() # Tambahkan .copy()

        if rider_df.empty:
            st.warning(f"Tidak ada data valid untuk {selected_rider} pada tahun {selected_year}.")
        else:
            # Filter hingga race ke-n
            rider_df_filtered = rider_df[rider_df['sequence'] <= selected_race].copy() # Tambahkan .copy()

            if rider_df_filtered.empty:
                st.warning(f"Tidak ada data hingga race ke-{selected_race}.")
            else:
                total_race = len(rider_df_filtered)
                avg_position = rider_df_filtered['position'].mean()
                consistency_score = rider_df_filtered['position'].std()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Balapan", total_race)
                col2.metric("Rata-rata Posisi", f"{avg_position:.2f}")
                col3.metric("Skor Konsistensi (Std Dev)", f"{consistency_score:.2f}")

                if len(rider_df_filtered) > 0:
                    best_race = rider_df_filtered.loc[rider_df_filtered['position'].idxmin()]
                    st.info(f"📌 Terbaik: Race ke-{int(best_race['sequence'])} (Posisi: {int(best_race['position'])})")

                    # Check for worst race, handle if only one race
                    if len(rider_df_filtered) > 1 and rider_df_filtered['position'].min() != rider_df_filtered['position'].max():
                        worst_race = rider_df_filtered.loc[rider_df_filtered['position'].idxmax()]
                        st.warning(f"📌 Terburuk: Race ke-{int(worst_race['sequence'])} (Posisi: {int(worst_race['position'])})")
                    elif len(rider_df_filtered) == 1:
                        st.info("Hanya ada satu data race, tidak ada perbandingan terbaik/terburuk.")
                    else:
                        st.info("Semua posisi sama, tidak ada posisi terburuk yang berbeda.")


                # Grafik Posisi
                st.subheader("Grafik Posisi")
                fig_pos = plt.figure(figsize=(10, 4))
                plt.plot(rider_df_filtered['sequence'], rider_df_filtered['position'], marker='o', color='teal', label='Posisi')
                plt.gca().invert_yaxis()
                plt.title(f"Grafik Posisi - {selected_rider} ({selected_year}) hingga Race ke-{selected_race}")
                plt.xlabel("Race Ke-")
                plt.ylabel("Posisi Finish")
                plt.grid(True)
                plt.legend()
                st.pyplot(fig_pos)

                # Grafik Poin
                st.subheader("Grafik Poin")
                fig_points = plt.figure(figsize=(10, 4))
                plt.plot(rider_df_filtered['sequence'], rider_df_filtered['points'], marker='s', color='orange', label='Poin')
                plt.title(f"Poin per Race - {selected_rider} ({selected_year}) hingga Race ke-{selected_race}")
                plt.xlabel("Race Ke-")
                plt.ylabel("Poin")
                plt.grid(True)
                plt.legend()
                st.pyplot(fig_points)

                st.subheader("📋 Posisi Aktual per Race:")
                for i, row in rider_df_filtered.iterrows():
                    st.write(f"Race ke-{int(row['sequence'])}: Posisi {int(row['position'])}")

                st.subheader("🔮 Prediksi LSTM:")
                if 'model' in locals() and 'scaler' in locals() and len(rider_df_filtered) >= time_step:
                    # Pastikan data terakhir memiliki dua kolom (posisi, poin)
                    last_n = rider_df_filtered[['position', 'points']].tail(time_step)
                    last_n_scaled = scaler.transform(last_n)
                    input_lstm = last_n_scaled.reshape(1, time_step, 2)

                    predicted_scaled = model.predict(input_lstm, verbose=0)
                    # Kita perlu dua kolom untuk inverse_transform, kolom kedua bisa nol
                    predicted_combined = np.hstack((predicted_scaled, np.zeros((predicted_scaled.shape[0], scaler.scale_.shape[0] - predicted_scaled.shape[1]))))
                    predicted_position = scaler.inverse_transform(predicted_combined)[0][0]

                    next_race = rider_df_filtered['sequence'].max() + 1
                    st.success(f"Estimasi Posisi Finish untuk Race ke-{next_race}: **{predicted_position:.2f}**")
                else:
                    st.info(f"❗ Data tidak cukup untuk prediksi LSTM (butuh minimal {time_step} race) atau model/scaler belum dimuat.")

                # --- PENTING: Tombol Ekspor ---
                st.subheader("Ekspor Data")
                filename = f"{selected_rider}_{selected_year}_Race{selected_race}_performance.csv"
                csv_data = rider_df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📤 Ekspor Data ke CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Unduh data performa pembalap yang difilter ke file CSV."
                )

    else:
        st.info("Silakan unggah file CSV dan pilih Pembalap, Tahun, dan Race Ke- untuk melihat dashboard.")
else:
    st.info("Silakan unggah file CSV Anda untuk memulai.")
