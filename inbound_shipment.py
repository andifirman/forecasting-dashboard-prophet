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
import plotly.graph_objects as go



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

# Ganti nilai negatif dengan angka acak antara 5 hingga 32
def replace_negative_with_random(data, min_val=5, max_val=32):
    """
    Fungsi untuk mengganti nilai negatif dengan angka acak dalam rentang tertentu (5-32).
    """
    return [random.randint(min_val, max_val) if value < 0 else value for value in data]


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

    # Validasi kolom
    if 'AREA 2' not in data.columns or 'Destname' not in data.columns or 'DATE' not in data.columns or 'Cnote' not in data.columns:
        return "Missing required columns in the dataset (AREA 2, Destname, DATE, Cnote).", 400

    # Preprocessing data
    data['ds'] = pd.to_datetime(data['DATE'])
    data.rename(columns={"Cnote": "y"}, inplace=True)

    # Menambahkan event khusus
    events = pd.DataFrame({
        'holiday': ['12.12', 'Hari Raya Natal'],
        'ds': [f"{year}-12-12", f"{year}-12-25"],
        'lower_window': [-days_before_event, -days_before_event],
        'upper_window': [days_after_event, days_after_event]
    })

    # Fungsi untuk melakukan forecasting berdasarkan AREA 2 dan Destname
    def forecast_by_area_and_dest(all_forecasting, events, year=2024):
        results = {}

        for area in all_forecasting['AREA 2'].unique():
            area_data = all_forecasting[all_forecasting['AREA 2'] == area]

            for dest in area_data['Destname'].unique():
                dest_data = area_data[area_data['Destname'] == dest]
                dest_data = dest_data.groupby('ds').agg({"y": "sum"}).reset_index()

                if dest_data.empty:
                    continue

                model = Prophet(holidays=events, changepoint_prior_scale=0.1)
                model.fit(dest_data)

                future = model.make_future_dataframe(periods=31)
                forecast = model.predict(future)
                forecast['week'] = forecast['ds'].dt.isocalendar().week

                for week, group in forecast.groupby('week'):
                    highest_value = group['yhat'].max()
                    highest_dates = group[group['yhat'] == highest_value]['ds']
                    second_highest_value = group[group['yhat'] < highest_value]['yhat'].max()

                    for event_date in ['2024-12-12', '2024-12-25']:
                        if pd.to_datetime(event_date).isocalendar().week == week:
                            forecast.loc[forecast['ds'] == event_date, 'yhat'] = highest_value

                            if second_highest_value is not None:
                                for date in highest_dates:
                                    if date != pd.to_datetime(event_date):
                                        forecast.loc[forecast['ds'] == date, 'yhat'] = second_highest_value

                            next_day = pd.to_datetime(event_date) + pd.Timedelta(days=1)
                            if next_day in forecast['ds'].values:
                                h1_value = highest_value * 0.98
                                forecast.loc[forecast['ds'] == next_day, 'yhat'] = min(
                                    h1_value, forecast.loc[forecast['ds'] == next_day, 'yhat'].values[0]
                                )

                december_forecast = forecast[(forecast['ds'] >= f"{year}-12-01") & (forecast['ds'] <= f"{year}-12-31")]
                total_forecast = december_forecast['yhat'].sum()

                results[(area, dest)] = {
                    'total_forecast': total_forecast,
                    'december_forecast': december_forecast
                }

        return results

    # Jalankan forecasting
    results = forecast_by_area_and_dest(data, events, year)

    # Format hasil ke dalam DataFrame untuk ditampilkan
    summary_data = []
    for (area, dest), result in results.items():
        december_forecast = result['december_forecast']
        total_forecast = result['total_forecast']

        summary_data.append({
            'AREA 2': area,
            'Destname': dest,
            'Total Forecast (Desember)': f"{total_forecast:,.0f}"
        })

    result_df = pd.DataFrame(summary_data)

    # Buat visualisasi grafik untuk All AREA 2 dan Destname
    fig = go.Figure()

    for (area, dest), result in results.items():
        december_forecast = result['december_forecast']
        fig.add_trace(go.Scatter(
            x=december_forecast['ds'],
            y=december_forecast['yhat'],
            mode='lines+markers',
            name=f"{area} - {dest}"
        ))

    fig.update_layout(
        title="Forecasted Shipments per AREA 2 and Destname (Desember 2024)",
        xaxis_title="Date",
        yaxis_title="Forecasted Shipments",
        legend_title="AREA 2 - Destname"
    )

    graph_html = fig.to_html(full_html=False)

    # Tampilkan hasil
    return render_template('result.html',
                           tables=[result_df.to_html(classes='table table-striped', index=False)],
                           graph_html=graph_html)



@app.route('/update-growth', methods=['POST'])
def update_growth():
    global result_df, base_forecast_df, forecast_data  # Akses base_forecast_df dan forecast_data

    if base_forecast_df is None or base_forecast_df.empty:
        return jsonify({'error': 'No base forecast data available. Please run analysis first.'}), 400

    try:
        # Ambil input growth dari pengguna
        growth = float(request.form['growth'])
    except ValueError:
        return jsonify({'error': 'Invalid growth value'}), 400

    try:
        # Pastikan data numerik diubah dengan benar
        base_forecast_df['Desember'] = pd.to_numeric(base_forecast_df['Desember'].replace({',': ''}, regex=True), errors='coerce')
        base_forecast_df['November'] = pd.to_numeric(base_forecast_df['November'].replace({',': ''}, regex=True), errors='coerce')

        # Menghitung ulang nilai Desember dan Growth, dengan pembulatan
        result_df['Desember'] = (base_forecast_df['Desember'] * (1 + growth / 100)).round()
        result_df['Growth %'] = ((result_df['Desember'] - base_forecast_df['November']) / base_forecast_df['November']) * 100

        # Format kolom untuk tampilan tabel
        result_df['Desember'] = result_df['Desember'].apply(lambda x: f"{int(x):,}")  # Format tanpa koma desimal
        result_df['Growth %'] = result_df['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")

        # Pastikan kolom `Initial Shipments` ada di forecast_data
        if 'Initial Shipments' not in forecast_data.columns:
            forecast_data['Initial Shipments'] = forecast_data['Forecasted Shipments']

        # Resetkan Forecasted Shipments ke nilai awal sebelum growth
        forecast_data['Forecasted Shipments'] = forecast_data['Initial Shipments'] * (1 + growth / 100)

        # Update `Forecasted Shipments` berdasarkan growth dengan pembulatan
        forecast_data['Forecasted Shipments'] = (forecast_data['Initial Shipments'] * (1 + growth / 100)).round()

        # Membuat grafik baru dengan Dropdown Menu
        fig = go.Figure()

        # Grafik untuk All Origin City
        all_cities_data = forecast_data.groupby('Date')['Forecasted Shipments'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=all_cities_data['Date'],
            y=all_cities_data['Forecasted Shipments'],
            mode='lines+markers',
            name='All Origin City',
            visible=True,  # Menampilkan grafik untuk All Origin City
            hovertemplate=("Date=%{x|%b %d, %Y}<br>"
                           "Total Forecasted Shipments=%{y:,}<extra></extra>")
        ))

        # Menambahkan trace untuk setiap kota
        for city in forecast_data['Origin City'].unique():
            city_data = forecast_data[forecast_data['Origin City'] == city]
            fig.add_trace(go.Scatter(
                x=city_data['Date'],
                y=city_data['Forecasted Shipments'],
                mode='lines+markers',
                name=city,
                visible=False,  # Semua trace di-hide awalnya
                customdata=city_data[['Origin City']],  # Tambahkan Origin City sebagai custom data
                hovertemplate=(
                    "Origin City=%{customdata[0]}<br>"
                    "Date=%{x|%b %d, %Y}<br>"
                    "Forecasted Shipments=%{y:,}<extra></extra>"
                )
            ))

        # Membuat dropdown menu
        buttons = []
        buttons.append(dict(label='All Origin Cities',
                            method='update',
                            args=[{'visible': [True] + [False] * len(forecast_data['Origin City'].unique())},
                                  {'title': "Forecasted Shipments for All Origin Cities"}]))

        for i, city in enumerate(forecast_data['Origin City'].unique()):
            visibility = [False] * (len(forecast_data['Origin City'].unique()) + 1)
            visibility[i + 1] = True  # Set hanya trace yang sesuai terlihat
            buttons.append(dict(
                label=city,
                method='update',
                args=[{'visible': visibility},
                      {'title': f"Forecasted Shipments for {city}"}]
            ))

        # Menambahkan menu dropdown ke layout
        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                x=0.5,  # Posisikan di tengah secara horizontal
                xanchor='center',
                y=1.02,  # Letakkan tepat di bawah judul
                yanchor='bottom'
            )],
            title="Forecasted Shipments per Origin City (Desember 2024)",
            xaxis_title="Date",
            yaxis_title="Forecasted Shipments",
            legend_title="Origin City",
            uniformtext_minsize=10,
            uniformtext_mode='hide',
            yaxis=dict(tickformat=',.0f')
        )

        # Konversi grafik ke HTML
        graph_html = fig.to_html(full_html=False)

        # Kirim data tabel dan grafik ke frontend
        updated_table = result_df.to_html(classes='table table-striped', index=False)
        return jsonify({'updated_table': updated_table, 'graph_html': graph_html})

    except Exception as e:
        # Tangani error yang terjadi
        return jsonify({'error': str(e)}), 500



@app.route('/compare-forecast-actual', methods=['POST'])
def compare_forecast_actual():
    global forecast_data, growth_data, actual_data

    # Gunakan hasil growth jika ada, jika tidak gunakan forecast_data
    comparison_data = growth_data if 'growth_data' in globals() and growth_data is not None else forecast_data

    # Pastikan forecast_data memiliki kolom datetime
    comparison_data['Date'] = pd.to_datetime(comparison_data['Date'], errors='coerce')

    try:
        # Unggah file data aktual
        actual_data_file = request.files['actual_data']
        if actual_data_file.filename == '':
            return jsonify({'error': "No file selected for actual data"}), 400

        if actual_data_file.filename.endswith('.csv'):
            actual_data = pd.read_csv(actual_data_file)
        elif actual_data_file.filename.endswith('.xlsx'):
            actual_data = pd.read_excel(actual_data_file)
        else:
            return jsonify({'error': "Invalid file format. Please upload a CSV or Excel file."}), 400
    except Exception as e:
        return jsonify({'error': f"Error reading actual data file: {e}"}), 400

    # Pastikan kolom 'DATE' dan 'Cnote' ada dalam data aktual
    if 'DATE' not in actual_data.columns or 'Cnote' not in actual_data.columns:
        return jsonify({'error': "Missing required columns in actual data file (DATE, Cnote)."}), 400

    # Transformasi data aktual
    actual_data['DATE'] = pd.to_datetime(actual_data['DATE'], errors='coerce')
    actual_data.rename(columns={"DATE": "Date", "Cnote": "Actual Shipments"}, inplace=True)

    # Kelompokkan data aktual berdasarkan Date dan Origin City
    actual_grouped = actual_data.groupby(['Date', 'Origin City'], as_index=False)['Actual Shipments'].sum()

    # Filter data aktual hanya untuk kota-kota yang ada di comparison_data
    actual_grouped = actual_grouped[actual_grouped['Origin City'].isin(comparison_data['Origin City'].unique())]

    # Gabungkan data forecasting dan aktual
    combined_data = pd.merge(
        comparison_data,
        actual_grouped,
        on=['Date', 'Origin City'],
        how='left'
    )

    # Membuat grafik dengan dropdown menu
    fig = go.Figure()

    # Tambahkan trace untuk "All Origin Cities"
    all_forecast = combined_data.groupby('Date')['Forecasted Shipments'].sum().reset_index()
    all_actual = combined_data.groupby('Date')['Actual Shipments'].sum().reset_index()

    fig.add_trace(go.Scatter(
        x=all_forecast['Date'],
        y=all_forecast['Forecasted Shipments'],
        mode='lines+markers',
        name='All Origin Cities (Forecast)',
        visible=True,
        hovertemplate=("Date=%{x|%b %d, %Y}<br>"
                       "Total Forecasted Shipments=%{y:,}<extra></extra>")
    ))

    if not all_actual.empty:
        fig.add_trace(go.Scatter(
            x=all_actual['Date'],
            y=all_actual['Actual Shipments'],
            mode='lines+markers',
            name='All Origin Cities (Actual)',
            visible=True,
            line=dict(dash='dot'),
            hovertemplate=("Date=%{x|%b %d, %Y}<br>"
                           "Total Actual Shipments=%{y:,}<extra></extra>")
        ))

    # Tambahkan trace untuk setiap Origin City
    for city in comparison_data['Origin City'].unique():
        city_forecast = combined_data[combined_data['Origin City'] == city]

        # Forecast trace untuk kota tertentu
        fig.add_trace(go.Scatter(
            x=city_forecast['Date'],
            y=city_forecast['Forecasted Shipments'],
            mode='lines+markers',
            name=f"{city} (Forecast)",
            visible=False,
            hovertemplate=("Origin City=%{name}<br>"
                           "Date=%{x|%b %d, %Y}<br>"
                           "Forecasted Shipments=%{y:,}<extra></extra>")
        ))

        # Actual trace untuk kota tertentu jika data tersedia
        if 'Actual Shipments' in city_forecast.columns:
            fig.add_trace(go.Scatter(
                x=city_forecast['Date'],
                y=city_forecast['Actual Shipments'],
                mode='lines+markers',
                name=f"{city} (Actual)",
                visible=False,
                line=dict(dash='dot'),
                hovertemplate=("Origin City=%{name}<br>"
                               "Date=%{x|%b %d, %Y}<br>"
                               "Actual Shipments=%{y:,}<extra></extra>")
            ))

    # Buat dropdown menu untuk memilih Origin City
    buttons = []
    # Tombol untuk "All Origin Cities"
    buttons.append(dict(
        label='All Origin Cities',
        method='update',
        args=[{'visible': [True, True] + [False] * (2 * len(comparison_data['Origin City'].unique()))},
              {'title': "Forecast vs Actual Shipments for All Origin Cities"}]
    ))

    # Tombol untuk setiap Origin City
    for i, city in enumerate(comparison_data['Origin City'].unique()):
        visibility = [False, False] + [False] * (2 * len(comparison_data['Origin City'].unique()))
        visibility[2 + 2 * i] = True  # Forecast trace untuk kota ini
        visibility[2 + 2 * i + 1] = True  # Actual trace untuk kota ini
        buttons.append(dict(
            label=city,
            method='update',
            args=[{'visible': visibility},
                  {'title': f"Forecast vs Actual Shipments for {city}"}]
        ))

    # Perbarui layout untuk dropdown menu
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.5,
            xanchor='center',
            y=1.15,
            yanchor='top'
        )],
        title="Forecast vs Actual Shipments for All Origin Cities",
        xaxis_title="Date",
        yaxis_title="Shipments",
        legend_title="Origin City",
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        yaxis=dict(tickformat=',.0f')
    )

    # Konversi grafik ke HTML
    graph_html = fig.to_html(full_html=False)

    # Kirim data tabel dan grafik ke frontend
    updated_table = result_df.to_html(classes='table table-striped', index=False)
    return jsonify({'updated_table': updated_table, 'graph_html': graph_html})


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)