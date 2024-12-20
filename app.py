# Library untuk pembacaan file dan template website
import os
from flask import Flask, render_template, request, redirect, url_for, send_file

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

@app.route('/')
def index():
    return render_template('index.html')  # Menampilkan halaman upload

@app.route('/analyze', methods=['POST'])
def analyze():
    # Proses upload file
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Validasi dan simpan file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Baca file yang diunggah
    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(filepath)
        else:
            return "Invalid file format. Please upload a CSV or Excel file.", 400
    except Exception as e:
        return f"Error reading file: {e}", 400

    # Ambil input user untuk parameter analisis
    try:
        days_before_event = int(request.form.get('days_before_event', 1))
        days_after_event = int(request.form.get('days_after_event', 6))
        growth_target = float(request.form.get('growth_target', 0))
    except ValueError:
        return "Invalid input for analysis parameters.", 400

    # Menggunakan data historis dari Januari hingga 1 Desember
    data['DATE'] = pd.to_datetime(data['DATE'])  # Pastikan kolom DATE dalam format datetime
    all_forecasting = data[data['DATE'] <= '2024-12-01']  # Data hingga 1 Desember 2024

    # Mengubah DATE menjadi ds dan Connote menjadi y untuk Prophet
    all_forecasting = all_forecasting.rename(columns={"DATE": "ds", "Connote": "y"})

    # Menambahkan event khusus
    events = pd.DataFrame({
        'holiday': ['12.12', 'Hari Raya Natal'],
        'ds': ['2024-12-12', '2024-12-25'],
        'lower_window': [-days_before_event, -days_before_event],
        'upper_window': [days_after_event, days_after_event]
    })

    # Fungsi forecasting
    def forecast_origin_city(city_name, all_forecasting, events, growth_target):
        city_data = all_forecasting[all_forecasting['Origin City'] == city_name]
        city_data = city_data.groupby('ds').agg({"y": "sum"}).reset_index()

        model = Prophet(holidays=events, changepoint_prior_scale=1)
        model.fit(city_data)

        future = model.make_future_dataframe(periods=31)
        forecast = model.predict(future)

        forecast['yhat'] = forecast['yhat'] * (1 + growth_target / 100)
        december_forecast = forecast[(forecast['ds'] >= '2024-12-01') & (forecast['ds'] <= '2024-12-31')]
        total_forecast = december_forecast['yhat'].sum()
        return total_forecast, december_forecast

    # Forecasting per Origin City
    origin_cities = all_forecasting['Origin City'].unique()
    forecast_results = {}
    for city in origin_cities:
        total_forecast, december_forecast = forecast_origin_city(city, all_forecasting, events, growth_target)
        forecast_results[city] = december_forecast

    # Data November
    november_data = data[(data['DATE'] >= '2024-11-01') & (data['DATE'] <= '2024-11-30')]
    total_november_per_city = november_data.groupby('Origin City')['Connote'].sum()

    # Gabungkan hasil ke DataFrame
    result_df = pd.DataFrame({
        'Origin City': sorted(origin_cities),
        'November': [total_november_per_city.get(city, 0) for city in sorted(origin_cities)],
        'Desember': [forecast_results[city]['yhat'].sum() for city in sorted(origin_cities)]
    })

    # Hitung Growth %
    result_df['Growth %'] = ((result_df['Desember'] - result_df['November']) / result_df['November']) * 100

    # Format angka untuk tampilan
    result_df['November'] = result_df['November'].apply(lambda x: f"{x:,.0f}")
    result_df['Desember'] = result_df['Desember'].apply(lambda x: f"{x:,.0f}")
    result_df['Growth %'] = result_df['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

    # Simpan hasil
    result_file = os.path.join(RESULT_FOLDER, 'forecast_results.csv')
    result_df.to_csv(result_file, index=False)

    # Hapus file upload
    os.remove(filepath)

    # Kirim hasil ke template
    return render_template('result.html', tables=[result_df.to_html(classes='table table-striped', index=False)])

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
