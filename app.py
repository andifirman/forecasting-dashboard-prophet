# Library untuk menangani semua aktivitas yang berkaitan dengan OS
# Seperti membaca dataset dari sebuah direktori 
import os

# Library untuk framework Flask dan beberapa utility yang dibutuhkan
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify

# Library untuk melakukan manipulasi terhadap dataset
import pandas as pd

# Library yang mengunggah model Prophet untuk melakukan forecasting
from prophet import Prophet

import random

# Library untuk visualisasi data
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt



app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Variabel global untuk menyimpan hasil forecast
forecast_results = {}
result_df = None
base_forecast_df = None

@app.route('/')
def index():
    return render_template('index.html')  # Menampilkan halaman upload

@app.route('/analyze', methods=['POST'])
def analyze():
    global result_df, base_forecast_df, forecast_data, total_december_forecast

    # Proses file upload
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Membaca file CSV/Excel
    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return "Invalid file format. Please upload a CSV or Excel file.", 400
    except Exception as e:
        return f"Error reading file: {e}", 400

    # Ambil input dari user
    try:
        year = int(request.form['year'])  # Tahun dari input pengguna
        days_before_event = int(request.form['days_before_event'])
        days_after_event = int(request.form['days_after_event'])
    except ValueError:
        return "Invalid input for year or days", 400

    # Preprocessing data
    data['DATE'] = pd.to_datetime(data['DATE'])
    all_forecasting = data[data['DATE'] <= '2024-12-01']
    all_forecasting = all_forecasting.rename(columns={"DATE": "ds", "Connote": "y"})

    # Menambahkan event khusus
    events = pd.DataFrame({
        'holiday': ['12.12', 'Hari Raya Natal'],
        'ds': [f"{year}-12-12", f"{year}-12-25"],
        'lower_window': [-days_before_event, -days_before_event],
        'upper_window': [days_after_event, days_after_event]
    })

    # Fungsi untuk forecasting per Origin City
    def forecast_origin_city(city_name, all_forecasting, events):
        city_data = all_forecasting[all_forecasting['Origin City'] == city_name]
        city_data = city_data.groupby('ds').agg({"y": "sum"}).reset_index()

        model = Prophet(holidays=events, changepoint_prior_scale=0.1)
        model.fit(city_data)

        future = model.make_future_dataframe(periods=31)
        forecast = model.predict(future)

        # Filter data Desember
        december_forecast = forecast[(forecast['ds'] >= f"{year}-12-01") & (forecast['ds'] <= f"{year}-12-31")]
        total_forecast = december_forecast['yhat'].sum()
        return total_forecast, december_forecast

    # Forecasting per Origin City
    origin_cities = all_forecasting['Origin City'].unique()
    results = {}
    december_forecasts = {}

    for city in origin_cities:
        total_forecast, december_forecast = forecast_origin_city(city, all_forecasting, events)
        results[city] = total_forecast
        december_forecasts[city] = december_forecast  # Simpan Desember forecast per kota

    # Total pengiriman bulan November per Origin City
    november_start = f"{year}-11-01"
    november_end = f"{year}-11-30"
    november_data = data[(data['DATE'] >= november_start) & (data['DATE'] <= november_end)]
    total_november_per_city = november_data.groupby('Origin City')['Connote'].sum()

    # Gabungkan hasil November dan Desember ke dalam DataFrame
    result_df = pd.DataFrame({
        'Origin City': sorted(origin_cities),
        'November': [total_november_per_city.get(city, 0) for city in sorted(origin_cities)],
        'Desember': [results[city] for city in sorted(origin_cities)]
    })

    # Hitung Growth %
    result_df['Growth %'] = ((result_df['Desember'] - result_df['November']) / result_df['November']) * 100

    # Terapkan batas minimum growth (-12%)
    growth_min = -12  # Minimum growth yang diizinkan

    def apply_growth_limit(row):
        if row['Growth %'] < growth_min:
            adjusted_forecast = row['November'] * (1 + growth_min / 100)
            return adjusted_forecast
        return row['Desember']

    result_df['Desember'] = result_df.apply(apply_growth_limit, axis=1)
    result_df['Growth %'] = ((result_df['Desember'] - result_df['November']) / result_df['November']) * 100

    # Format angka agar lebih rapi
    result_df['November'] = result_df['November'].apply(lambda x: f"{x:,.0f}")
    result_df['Desember'] = result_df['Desember'].apply(lambda x: f"{x:,.0f}")
    result_df['Growth %'] = result_df['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

    # Simpan baseline data (numerik) sebelum diformat
    base_forecast_df = result_df.copy()

    # Total pengiriman bulan Desember berdasarkan forecasting
    total_december_forecast = sum(results.values())

    # Menggabungkan data forecasting dari semua Origin City
    forecast_data = pd.DataFrame()
    for city, df in december_forecasts.items():
        df['Origin City'] = city
        forecast_data = pd.concat([forecast_data, df])

    # Format hasil agar lebih rapi
    forecast_data['yhat'] = forecast_data['yhat'].apply(lambda x: round(x))
    forecast_data = forecast_data.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Shipments'})

    # Total forecast per Origin City berdasarkan forecast harian Desember
    total_forecast_per_city = forecast_data.groupby('Origin City')['Forecasted Shipments'].sum().reset_index()
    total_forecast_per_city = total_forecast_per_city.rename(columns={'Forecasted Shipments': 'Total Forecasted Shipments'})

    # Menggabungkan informasi jumlah forecast dengan data forecast
    forecast_data = pd.merge(forecast_data, total_forecast_per_city, on='Origin City', how='left')

    # Menampilkan jumlah forecast per Origin City
    print(total_forecast_per_city)

    # Membuat visualisasi Line Graph berdasarkan Tanggal dan Origin City
    fig = px.line(
        forecast_data,
        x="Date",
        y="Forecasted Shipments",
        color="Origin City",
        markers=True
    )

    # Menghilangkan teks yang muncul di sepanjang garis
    fig.update_traces(text=None)

    # Menambahkan layout untuk mempercantik grafik
    fig.update_layout(
        title="Forecasted Shipments per Origin City (Desember 2024)",
        xaxis_title="Date",
        yaxis_title="Forecasted Shipments",
        legend_title="Origin City",
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        yaxis=dict(
            tickformat=',.0f'  # Format angka dengan pemisah ribuan
        )
    )

    # Menampilkan grafik di halaman web Flask
    graph_html = fig.to_html(full_html=False)

    # Tampilkan hasil
    return render_template('result.html', \
                           tables=[result_df.to_html(classes='table table-striped', index=False)],\
                           total_december_forecast=f"{total_december_forecast:,.0f}", \
                           graph_html=graph_html)



@app.route('/update-growth', methods=['POST'])
def update_growth():
    global result_df, base_forecast_df, forecast_data  # Akses base_forecast_df dan forecast_data

    if base_forecast_df is None or base_forecast_df.empty:
        return jsonify({'error': 'No base forecast data available. Please run analysis first.'}), 400

    try:
        growth = float(request.form['growth'])
    except ValueError:
        return jsonify({'error': 'Invalid growth value'}), 400

    try:
        # Pastikan data numeric diubah dengan benar
        base_forecast_df['Desember'] = pd.to_numeric(base_forecast_df['Desember'].replace({',': ''}, regex=True), errors='coerce')
        base_forecast_df['November'] = pd.to_numeric(base_forecast_df['November'].replace({',': ''}, regex=True), errors='coerce')

        # Menghitung Desember dan Growth
        result_df['Desember'] = base_forecast_df['Desember'] * (1 + growth / 100)
        result_df['Growth %'] = ((result_df['Desember'] - base_forecast_df['November']) / base_forecast_df['November']) * 100

        result_df['Desember'] = result_df['Desember'].apply(lambda x: f"{x:,.0f}")
        result_df['Growth %'] = result_df['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

        # Pastikan Initial Shipments ada di forecast_data
        if 'Initial Shipments' not in forecast_data.columns:
            forecast_data['Initial Shipments'] = forecast_data['Forecasted Shipments']

        # Resetkan Forecasted Shipments ke nilai awal sebelum growth
        forecast_data['Forecasted Shipments'] = forecast_data['Initial Shipments'] * (1 + growth / 100)

        # Membuat grafik baru
        fig = px.line(
            forecast_data,
            x="Date",
            y="Forecasted Shipments",
            color="Origin City",
            markers=True
        )

        fig.update_traces(text=None)
        fig.update_layout(
            title="Forecasted Shipments per Origin City (Desember 2024)",
            xaxis_title="Date",
            yaxis_title="Forecasted Shipments",
            legend_title="Origin City",
            uniformtext_minsize=10,
            uniformtext_mode='hide',
            yaxis=dict(tickformat=',.0f')
        )

        # Kirim tabel yang diperbarui dan grafik baru
        updated_table = result_df.to_html(classes='table table-striped', index=False)
        graph_html = fig.to_html(full_html=False)
        return jsonify({'updated_table': updated_table, 'graph_html': graph_html})

    except Exception as e:
        # Menampilkan error yang lebih spesifik
        return jsonify({'error': str(e)}), 500




@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)