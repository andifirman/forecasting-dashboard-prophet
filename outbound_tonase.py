# Import libraries
import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from prophet import Prophet
import random
import plotly.graph_objects as go

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Global variables
forecast_results = {}
result_df = None
base_forecast_df = None

@app.route('/')
def index():
    return render_template('index.html')  # Display upload page

# Replace negative values with random numbers
def replace_negative_with_random(data, min_val=5, max_val=32):
    return [random.randint(min_val, max_val) if value < 0 else value for value in data]

@app.route('/analyze', methods=['POST'])
def analyze():
    global result_df, base_forecast_df, forecast_results

    # File upload handling
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Read dataset
    try:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return "Invalid file format. Please upload a CSV or Excel file.", 400
    except Exception as e:
        return f"Error reading file: {e}", 400

    # Get user input
    try:
        year = int(request.form['year'])
        days_before_event = int(request.form['days_before_event'])
        days_after_event = int(request.form['days_after_event'])
    except ValueError:
        return "Invalid input for year or days", 400

    # Process data for both Connote and Weight
    results = {}
    for column in ['Connote', 'Weight']:
        column_data = data.rename(columns={"DATE": "ds", column: "y"})
        column_data['ds'] = pd.to_datetime(column_data['ds'])
        column_forecast = column_data[column_data['ds'] <= f'{year}-12-01']

        # Define special events
        events = pd.DataFrame({
            'holiday': ['12.12', 'Hari Raya Natal'],
            'ds': [f"{year}-12-12", f"{year}-12-25"],
            'lower_window': [-days_before_event, -days_before_event],
            'upper_window': [days_after_event, days_after_event]
        })

        # Forecasting function
        def forecast_origin_city(city_name, data, events, year):
            city_data = data[data['Origin City'] == city_name].groupby('ds').agg({"y": "sum"}).reset_index()
            model = Prophet(holidays=events, changepoint_prior_scale=0.1)
            model.fit(city_data)
            future = model.make_future_dataframe(periods=31)
            forecast = model.predict(future)
            forecast['week'] = forecast['ds'].dt.isocalendar().week
            december_forecast = forecast[(forecast['ds'] >= f"{year}-12-01") & (forecast['ds'] <= f"{year}-12-31")]
            total_forecast = december_forecast['yhat'].sum()
            return total_forecast, december_forecast

        # Perform forecasting
        origin_cities = column_forecast['Origin City'].unique()
        city_results = {}
        december_forecasts = {}

        for city in origin_cities:
            total_forecast, december_forecast = forecast_origin_city(city, column_forecast, events, year)
            city_results[city] = total_forecast
            december_forecasts[city] = december_forecast

        # Prepare result dataframes
        november_data = data[(data['ds'] >= f"{year}-11-01") & (data['ds'] <= f"{year}-11-30")]
        november_total = november_data.groupby('Origin City')[column].sum()
        result_df = pd.DataFrame({
            'Origin City': sorted(origin_cities),
            'November': [november_total.get(city, 0) for city in sorted(origin_cities)],
            'Desember': [city_results[city] for city in sorted(origin_cities)]
        })

        result_df['Desember'] = replace_negative_with_random(result_df['Desember'])
        result_df['Growth %'] = ((result_df['Desember'] - result_df['November']) / result_df['November']) * 100
        result_df['Desember'] = replace_negative_with_random(result_df['Desember'])
        base_forecast_df = result_df.copy()

        forecast_data = pd.DataFrame()
        for city, df in december_forecasts.items():
            df['Origin City'] = city
            forecast_data = pd.concat([forecast_data, df])
        forecast_data = forecast_data.rename(columns={'ds': 'Date', 'yhat': f'Forecasted {column}'})
        results[column] = {
            'result_df': result_df,
            'forecast_data': forecast_data
        }

    forecast_results = results

    return jsonify({'status': 'success', 'message': 'Forecasting completed'})

@app.route('/visualize', methods=['GET'])
def visualize():
    column = request.args.get('column', 'Connote')
    if column not in forecast_results:
        return jsonify({'error': 'Invalid column selected'}), 400

    # Get data for visualization
    result_df = forecast_results[column]['result_df']
    forecast_data = forecast_results[column]['forecast_data']

    # Create line graph
    fig = go.Figure()
    all_cities_data = forecast_data.groupby('Date').sum().reset_index()
    fig.add_trace(go.Scatter(
        x=all_cities_data['Date'],
        y=all_cities_data[f'Forecasted {column}'],
        mode='lines+markers',
        name='All Origin City'
    ))

    for city in forecast_data['Origin City'].unique():
        city_data = forecast_data[forecast_data['Origin City'] == city]
        fig.add_trace(go.Scatter(
            x=city_data['Date'],
            y=city_data[f'Forecasted {column}'],
            mode='lines+markers',
            name=city
        ))

    fig.update_layout(
        title=f"Forecasted {column} per Origin City",
        xaxis_title="Date",
        yaxis_title=f"Forecasted {column}",
        legend_title="Origin City"
    )

    graph_html = fig.to_html(full_html=False)
    return render_template('result.html', tables=[result_df.to_html(classes='table')], graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
