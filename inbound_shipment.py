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
    # Fungsi untuk mengganti nilai negatif dengan angka acak dalam rentang tertentu (5-32).
    return [random.randint(min_val, max_val) if value < 0 else value for value in data]

# Fungsi untuk mengganti komponen desimal growth yang -5.00%
def adjust_growth_decimal(value):
    # Jika growth % adalah -5.00%, maka ubah bagian desimalnya menjadi angka antara 0.08 sampai 0.97
    if value == -5.00:
        return round(value + random.uniform(0.08, 0.97), 2)
    return value




# --- 1. Fungsi Forecasting yang Telah Digabungkan ---

def forecast_group(group_data, events, periods=31):
    """
    Melakukan forecasting menggunakan Prophet untuk data yang diberikan.
    
    Parameters:
    - group_data: DataFrame dengan kolom 'ds' dan 'y'.
    - events: DataFrame events untuk Prophet.
    - periods: Jumlah hari ke depan untuk forecasting.
    
    Returns:
    - DataFrame hasil forecast dengan kolom tambahan 'week'.
    """
    # Inisialisasi dan melatih model Prophet
    model = Prophet(holidays=events, changepoint_prior_scale=0.1)
    model.fit(group_data)
    
    # Membuat dataframe future
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Tambahkan kolom 'week' untuk identifikasi mingguan
    forecast['week'] = forecast['ds'].dt.isocalendar().week
    
    # Proses setiap minggu untuk menyesuaikan jika ada event
    for week, group in forecast.groupby('week'):
        # Identifikasi nilai tertinggi dan tanggalnya
        highest_value = group['yhat'].max()
        highest_dates = group[group['yhat'] == highest_value]['ds']
        
        # Cari nilai tertinggi kedua
        second_highest_value = group[group['yhat'] < highest_value]['yhat'].max()
        
        # Cek apakah ada event di minggu ini
        for event_date in events['ds']:
            if pd.to_datetime(event_date).isocalendar().week == week:
                event_date_dt = pd.to_datetime(event_date)
                if event_date_dt in forecast['ds'].values:
                    # Set nilai tertinggi ke tanggal event
                    forecast.loc[forecast['ds'] == event_date_dt, 'yhat'] = highest_value
                    
                    # Update nilai tertinggi sebelumnya menjadi nilai tertinggi kedua
                    if pd.notnull(second_highest_value):
                        for date in highest_dates:
                            if date != event_date_dt:
                                forecast.loc[forecast['ds'] == date, 'yhat'] = second_highest_value
                    
                    # Turunkan nilai H+1 sebesar 2%
                    next_day = event_date_dt + pd.Timedelta(days=1)
                    if next_day in forecast['ds'].values:
                        h1_value = highest_value * 0.98  # Turunkan 2%
                        forecast.loc[forecast['ds'] == next_day, 'yhat'] = min(
                            h1_value, forecast.loc[forecast['ds'] == next_day, 'yhat'].values[0]
                        )
    
    # Filter data untuk bulan Desember
    december_forecast = forecast[(forecast['ds'] >= f"{events['ds'].dt.year.unique()[0]}-12-01") & 
                                 (forecast['ds'] <= f"{events['ds'].dt.year.unique()[0]}-12-31")]
    
    # Kembalikan dataframe forecast Desember
    return december_forecast

def forecast_per_area_area2_destname(shipment_forecasting, events):
    """
    Melakukan forecasting per kombinasi AREA, AREA 2, dan Destname.
    
    Returns:
    - DataFrame gabungan forecast per AREA, AREA 2, Destname.
    """
    # Rename columns for Prophet
    shipment_forecasting_group = shipment_forecasting.rename(columns={"DATE": "ds", "Cnote": "y"})
    
    # Mendapatkan unique combinations
    combinations = shipment_forecasting_group[['AREA', 'AREA 2', 'Destname']].drop_duplicates()
    
    december_forecasts = {}
    
    for idx, row in combinations.iterrows():
        area = row['AREA']
        area2 = row['AREA 2']
        destname = row['Destname']
        
        # Filter data per kombinasi
        group_data = shipment_forecasting_group[
            (shipment_forecasting_group['AREA'] == area) &
            (shipment_forecasting_group['AREA 2'] == area2) &
            (shipment_forecasting_group['Destname'] == destname)
        ]
        
        # Group dan sum 'y' per tanggal
        group_data = group_data.groupby('ds').agg({"y": "sum"}).reset_index()
        
        # Forecast
        december_forecast = forecast_group(group_data, events)
        
        # Tambahkan informasi grup
        december_forecast['AREA'] = area
        december_forecast['AREA 2'] = area2
        december_forecast['Destname'] = destname
        
        # Simpan ke dictionary
        key = (area, area2, destname)
        december_forecasts[key] = december_forecast
    
    # Gabungkan semua forecast
    forecast_data = pd.DataFrame()
    for key, df in december_forecasts.items():
        forecast_data = pd.concat([forecast_data, df], ignore_index=True)
    
    # Format hasil agar lebih rapi
    forecast_data['Forecasted Shipments'] = forecast_data['yhat'].apply(lambda x: round(x))
    forecast_data = forecast_data.rename(columns={'ds': 'Date'})
    
    return forecast_data

def forecast_per_area(shipment_forecasting, events):
    """
    Melakukan forecasting per AREA dengan mengagregasi data dari AREA 2 dan Destname.
    
    Returns:
    - DataFrame gabungan forecast per AREA.
    """
    # Rename columns for Prophet
    shipment_forecasting_area = shipment_forecasting.rename(columns={"DATE": "ds", "Cnote": "y"})
    
    # Mendapatkan unique AREA
    areas = shipment_forecasting_area['AREA'].unique()
    
    december_forecasts = {}
    
    for area in areas:
        # Filter data per AREA
        area_data = shipment_forecasting_area[shipment_forecasting_area['AREA'] == area]
        
        # Group dan sum 'y' per tanggal
        area_data = area_data.groupby('ds').agg({"y": "sum"}).reset_index()
        
        # Forecast
        december_forecast = forecast_group(area_data, events)
        
        # Tambahkan informasi AREA
        december_forecast['AREA'] = area
        december_forecast['AREA 2'] = None
        december_forecast['Destname'] = None
        
        # Simpan ke dictionary
        december_forecasts[area] = december_forecast
    
    # Gabungkan semua forecast
    forecast_data = pd.DataFrame()
    for area, df in december_forecasts.items():
        forecast_data = pd.concat([forecast_data, df], ignore_index=True)
    
    # Format hasil agar lebih rapi
    forecast_data['Forecasted Shipments'] = forecast_data['yhat'].apply(lambda x: round(x))
    forecast_data = forecast_data.rename(columns={'ds': 'Date'})
    
    return forecast_data




# --- 3. Endpoint /analyze yang Dimodifikasi ---
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # --- 3.1. Proses File Upload ---
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

        # --- 3.2. Ambil Input dari User ---
        try:
            year = int(request.form['year'])  # Tahun dari input pengguna
            days_before_event = int(request.form['days_before_event'])
            days_after_event = int(request.form['days_after_event'])
        except (ValueError, KeyError):
            return "Invalid input for year or days", 400

        # --- 3.3. Preprocessing Data ---
        data['DATE'] = pd.to_datetime(data['DATE'])
        shipment_forecasting = data[data['DATE'] <= f'{year}-12-01']
        shipment_forecasting = shipment_forecasting.rename(columns={"DATE": "ds", "Cnote": "y"})

        # --- 3.4. Menambahkan Event Khusus ---
        events = pd.DataFrame({
            'holiday': ['12.12', 'Hari Raya Natal'],
            'ds': [f"{year}-12-12", f"{year}-12-25"],
            'lower_window': [-days_before_event, -days_before_event],  # Hari sebelum event
            'upper_window': [days_after_event, days_after_event]      # Hari setelah event
        })

        # --- 3.5. Forecasting per Group dan per AREA ---
        # Forecast per AREA, AREA 2, Destname
        forecast_per_group = forecast_per_area_area2_destname(data, events)
        
        # Forecast per AREA
        forecast_per_area_df = forecast_per_area(data, events)
        
        # --- 3.6. Menggabungkan Hasil Forecasting ---
        # Forecast per Group
        forecast_per_group_grouped = forecast_per_group.groupby(['AREA', 'AREA 2', 'Destname'])['Forecasted Shipments'].sum().reset_index()
        
        # Forecast per AREA
        forecast_per_area_grouped = forecast_per_area_df.groupby('AREA')['Forecasted Shipments'].sum().reset_index()
        
        # --- 3.7. Mengambil Data November ---
        november_data_group = data[
            (data['DATE'] >= f"{year}-11-01") & 
            (data['DATE'] <= f"{year}-11-30")
        ].groupby(['AREA', 'AREA 2', 'Destname'])['Cnote'].sum().reset_index()
        
        november_data_area = data[
            (data['DATE'] >= f"{year}-11-01") & 
            (data['DATE'] <= f"{year}-11-30")
        ].groupby(['AREA'])['Cnote'].sum().reset_index()

        # --- 3.8. Menggabungkan Data November dan Forecast Desember per Group ---
        result_df_group = pd.merge(november_data_group, forecast_per_group_grouped, on=['AREA', 'AREA 2', 'Destname'], how='left')
        result_df_group = result_df_group.rename(columns={'Cnote': 'November', 'Forecasted Shipments': 'Desember'})
        
        # Hitung Growth %
        result_df_group['Growth %'] = ((result_df_group['Desember'] - result_df_group['November']) / result_df_group['November']) * 100
        
        # --- 3.9. Menggabungkan Data November dan Forecast Desember per AREA ---
        result_df_area = pd.merge(november_data_area, forecast_per_area_grouped, on='AREA', how='left')
        result_df_area = result_df_area.rename(columns={'Cnote': 'November', 'Forecasted Shipments': 'Desember'})
        
        # Hitung Growth %
        result_df_area['Growth %'] = ((result_df_area['Desember'] - result_df_area['November']) / result_df_area['November']) * 100
        
        # --- 3.10. Penyesuaian Growth % dan Forecast ---
        # Fungsi untuk menyesuaikan growth dan menambah angka acak pada desimal
        def adjust_forecast_growth(november, december):
            if november == 0:
                growth_percentage = 0
            else:
                growth_percentage = ((december - november) / november) * 100
            
            # Cek apakah growth lebih kecil dari -5%
            if growth_percentage < -5.00:
                growth_percentage = -5.00
                # Sesuaikan Desember untuk menyesuaikan growth di -5%
                december = november * (1 + growth_percentage / 100)
            
            # Tambahkan angka acak di belakang koma menggunakan fungsi replace_negative_with_random
            random_decimal = replace_negative_with_random([0])[0] / 100  # Hanya mengambil nilai pertama, acak antara 5-32
            december = round(december + random_decimal, 2)
    
            # Adjust decimal part of growth if it's exactly -5.00%
            growth_percentage = adjust_growth_decimal(growth_percentage)
    
            return december, growth_percentage

        # Terapkan penyesuaian pada result_df_group
        result_df_group[['Desember', 'Growth %']] = result_df_group.apply(
            lambda row: pd.Series(adjust_forecast_growth(row['November'], row['Desember'])), axis=1
        )
        
        # Terapkan penyesuaian pada result_df_area
        result_df_area[['Desember', 'Growth %']] = result_df_area.apply(
            lambda row: pd.Series(adjust_forecast_growth(row['November'], row['Desember'])), axis=1
        )
        
        # --- 3.11. Format Angka Agar Lebih Rapi ---
        # Format untuk Group
        result_df_group['November'] = result_df_group['November'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        result_df_group['Desember'] = result_df_group['Desember'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        result_df_group['Growth %'] = result_df_group['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        
        # Format untuk AREA
        result_df_area['November'] = result_df_area['November'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        result_df_area['Desember'] = result_df_area['Desember'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        result_df_area['Growth %'] = result_df_area['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        
        # --- 3.12. Membuat Tabel HTML ---
        area_table = result_df_area.to_html(classes='table table-striped table-hover', index=False)
        breakdown_table = result_df_group.to_html(classes='table table-striped table-hover', index=False)
        
        # --- 3.13. Membuat Visualisasi Line Graph Berdasarkan Tanggal dan AREA ---
        fig = go.Figure()
        
        # Data untuk All AREA
        all_area_data = forecast_per_area_df.groupby('Date')['Forecasted Shipments'].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=all_area_data['Date'],
            y=all_area_data['Forecasted Shipments'],
            mode='lines+markers',
            name='All Areas',
            visible=True
        ))
        
        # Data untuk masing-masing AREA
        for area in result_df_area['AREA']:
            if area == 'Total':
                continue  # Skip total if exists
            area_data = forecast_per_area_df[forecast_per_area_df['AREA'] == area].groupby('Date')['Forecasted Shipments'].sum().reset_index()
            fig.add_trace(go.Scatter(
                x=area_data['Date'],
                y=area_data['Forecasted Shipments'],
                mode='lines+markers',
                name=area,
                visible=False
            ))
        
        # Dropdown menu
        buttons = []
        buttons.append(dict(label='All Areas',
                            method='update',
                            args=[{'visible': [True] + [False] * (len(result_df_area['AREA']) - 1)},
                                  {'title': "Forecast Summary for All Areas"}]))
        
        for i, area in enumerate(result_df_area['AREA']):
            visibility = [False] * (len(result_df_area['AREA']))
            visibility[i] = True
            buttons.append(dict(label=area,
                                method='update',
                                args=[{'visible': visibility},
                                      {'title': f"Forecast Summary for {area}"}]))
        
        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                x=0.5,
                xanchor='center',
                y=1.15,
                yanchor='top'
            )],
            title="Forecast Summary by AREA",
            xaxis_title="Date",
            yaxis_title="Forecasted Shipments",
            legend_title="AREA",
            uniformtext_minsize=10,
            uniformtext_mode='hide',
            yaxis=dict(tickformat=',.0f')
        )
        
        graph_html = fig.to_html(full_html=False)
        
        # --- 3.14. Menambahkan Baris Total ---
        # Hitung total November dan Desember untuk AREA
        total_november = data[
            (data['DATE'] >= f"{year}-11-01") & 
            (data['DATE'] <= f"{year}-11-30")
        ]['Cnote'].sum()
        total_desember = forecast_per_area_grouped['Forecasted Shipments'].sum()
        
        # Hitung Growth % untuk total
        if total_november != 0:
            total_growth = ((total_desember - total_november) / total_november) * 100
        else:
            total_growth = None
        
        # Buat DataFrame untuk total
        total_row = pd.DataFrame({
            'AREA': ['Total'],
            'November': [total_november],
            'Desember': [total_desember],
            'Growth %': [total_growth]
        })
        
        # Format angka untuk total_row
        total_row['November'] = total_row['November'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        total_row['Desember'] = total_row['Desember'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
        total_row['Growth %'] = total_row['Growth %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
        
        # Gabungkan dengan result_df_area
        result_df_area = pd.concat([result_df_area, total_row], ignore_index=True)
        
        # Update area_table dengan total
        area_table = result_df_area.to_html(classes='table table-striped table-hover', index=False)
        
        # --- 3.15. Render Template dengan Hasil ---
        return render_template(
            'resultin.html',
            area_table=area_table,
            breakdown_table=breakdown_table,
            graph_html=graph_html
        )
    
    except Exception as e:
        return f"An error occurred: {e}", 500



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