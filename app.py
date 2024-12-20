# Library untuk pembacaan file dan template website
import os
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify

# Library untuk melakukan analisis dan forecasting
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import matplotlib.dates as mdates


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)



# Variabel global untuk menyimpan hasil forecast
forecast_results = {}
result_df = None




@app.route('/')
def index():
    return render_template('index.html')  # Menampilkan halaman upload

@app.route('/analyze', methods=['POST'])
def analyze():
    global result_df, base_forecast_df  # Tambahkan base_forecast_df

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
    start_date = f"{year}-12-01"
    end_date = f"{year}-12-31"
    all_forecasting = data[data['DATE'] <= start_date]
    all_forecasting = all_forecasting.rename(columns={"DATE": "ds", "Connote": "y"})

    # Event khusus
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

        model = Prophet(holidays=events, changepoint_prior_scale=1)
        model.fit(city_data)

        future = model.make_future_dataframe(periods=31)
        forecast = model.predict(future)

        # Data Desember
        december_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
        return december_forecast

    # Forecast semua Origin City
    origin_cities = all_forecasting['Origin City'].unique()
    forecast_results = {}
    for city in origin_cities:
        forecast_results[city] = forecast_origin_city(city, all_forecasting, events)

    # Data bulan November
    november_start = f"{year}-11-01"
    november_end = f"{year}-11-30"
    november_data = data[(data['DATE'] >= november_start) & (data['DATE'] <= november_end)]
    total_november_per_city = november_data.groupby('Origin City')['Connote'].sum()

    # Gabungkan hasil November dan Desember
    result_df = pd.DataFrame({
        'Origin City': sorted(origin_cities),
        'November': [total_november_per_city.get(city, 0) for city in sorted(origin_cities)],
        'Desember': [forecast_results[city]['yhat'].sum() for city in sorted(origin_cities)]
    })
    result_df['Growth %'] = ((result_df['Desember'] - result_df['November']) / result_df['November']) * 100

    # Simpan baseline data (numerik) sebelum diformat
    base_forecast_df = result_df.copy()

    # Format ulang untuk tampilan
    result_df['November'] = result_df['November'].apply(lambda x: f"{x:,.0f}")
    result_df['Desember'] = result_df['Desember'].apply(lambda x: f"{x:,.0f}")
    result_df['Growth %'] = result_df['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

    return render_template('result.html', tables=[result_df.to_html(classes='table table-striped', index=False)])


@app.route('/update-growth', methods=['POST'])
def update_growth():
    global result_df, base_forecast_df  # Akses base_forecast_df

    # Validasi: Pastikan base_forecast_df sudah diinisialisasi
    if base_forecast_df is None or base_forecast_df.empty:
        return jsonify({'error': 'No base forecast data available. Please run analysis first.'}), 400

    try:
        growth = float(request.form['growth'])
        print(f"Growth value received: {growth}")  # Debugging
    except ValueError:
        return jsonify({'error': 'Invalid growth value'}), 400

    try:
        # Gunakan data asli (numerik) dari base_forecast_df
        result_df['Desember'] = base_forecast_df['Desember'] * (1 + growth / 100)
        result_df['Growth %'] = ((result_df['Desember'] - base_forecast_df['November']) /
                                 base_forecast_df['November']) * 100

        # Format ulang untuk tampilan
        result_df['Desember'] = result_df['Desember'].apply(lambda x: f"{x:,.0f}")
        result_df['Growth %'] = result_df['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

        # Kirim tabel yang diperbarui ke browser
        updated_table = result_df.to_html(classes='table table-striped', index=False)
        return jsonify({'updated_table': updated_table})
    except Exception as e:
        print(f"Error while updating growth: {e}")  # Debug log
        return jsonify({'error': 'Internal Server Error'}), 500

    # Kirim data hasil analisis ke template
    # return render_template('result.html', tables=[result_df.to_html(classes='table table-striped', index=False).strip().replace('\n', '')])
    
    
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
