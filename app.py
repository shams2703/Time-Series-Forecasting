import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Data loading function with date parsing
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file, parse_dates=True, infer_datetime_format=True)
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_datetime(data[col])
            except Exception:
                continue
    return data

# Data preparation function with date column detection
def prepare_data(data, date_column, target_column):
    try:
        data[date_column] = pd.to_datetime(data[date_column])
    except Exception as e:
        st.error(f"Error converting {date_column} to datetime format: {e}")
        return None
    data_prepared = data[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'})
    return data_prepared

# Forecasting function
def forecast_data(model, forecast_period, time_unit):
    future = model.make_future_dataframe(periods=forecast_period, freq=time_unit[0].upper())
    future = future[future['ds'] > model.history['ds'].max()]
    forecast = model.predict(future)
    return forecast

# Hindcasting function
def hindcast_data(model, df, time_unit, hindcast_period):
    hindcast_end = df['ds'].max()
    if time_unit == "Days":
        hindcast_start = hindcast_end - pd.to_timedelta(hindcast_period, unit='d')
    elif time_unit == "Weeks":
        hindcast_start = hindcast_end - pd.to_timedelta(hindcast_period * 7, unit='d')
    elif time_unit == "Months":
        hindcast_start = hindcast_end - pd.DateOffset(months=hindcast_period)
    elif time_unit == "Years":
        hindcast_start = hindcast_end - pd.DateOffset(years=hindcast_period)

    hindcast_df = df[(df['ds'] >= hindcast_start) & (df['ds'] <= hindcast_end)]
    hindcast = model.predict(hindcast_df[['ds']])
    return hindcast_df, hindcast

# Anomaly detection using IQR method
def detect_anomalies(data):
    q1 = data['y'].quantile(0.25)
    q3 = data['y'].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    data['anomaly'] = np.where((data['y'] < lower_bound) | (data['y'] > upper_bound), 1, 0)
    return data

# Plotting functions
def plot_forecast(actual_df, forecast):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_df['ds'], actual_df['y'], label='Actual Data', color='black')
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)
    ax.set_title('Future Forecast of the Time Series')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

def plot_forecast_components(model, forecast):
    fig = model.plot_components(forecast)
    fig.suptitle('Forecast Components', y=1.02)
    st.pyplot(fig)

def plot_hindcast(actual_df, hindcast_df, hindcast):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual_df['ds'], actual_df['y'], label='Actual Data', color='black')
    ax.plot(hindcast['ds'], hindcast['yhat'], label='Hindcast', color='red')
    ax.fill_between(hindcast['ds'], hindcast['yhat_lower'], hindcast['yhat_upper'], color='lightcoral', alpha=0.5)
    ax.set_title('Hindcast of the Time Series')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

def plot_anomalies(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['ds'], data['y'], label='Actual Data', color='black')
    anomalies = data[data['anomaly'] == 1]
    ax.scatter(anomalies['ds'], anomalies['y'], color='red', label='Anomalies', marker='o')
    ax.set_title('Anomaly Detection')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)

# Main function to run the Streamlit app
def run_app():
    st.title("Time Series Forecasting with Hindcasting and Anomaly Detection")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        
        # Show first five rows of the dataset
        st.write("First five rows of the dataset:")
        st.dataframe(data.head())

        # Detect date column or prompt user to select one
        date_columns = data.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        if not date_columns:
            st.write("No automatic date column detected. Please select the date column.")
            date_column = st.selectbox("Select Date Column", data.columns)
        else:
            date_column = date_columns[0]
            st.write(f"Detected Date Column: {date_column}")

        # Filter target columns excluding the date column
        target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if date_column in target_columns:
            target_columns.remove(date_column)
        
        target_column = st.selectbox("Select Target Column", target_columns)
        
        # Select time unit
        time_unit = st.selectbox("Select Time Unit", ["Days", "Weeks", "Months", "Years"])

        # Forecasting and Hindcasting parameters
        forecast_period = st.number_input("Forecast Period (number of units)", min_value=1, value=30)
        hindcast_period = st.number_input("Hindcast Period (number of units)", min_value=1, value=30)
        confidence_interval = st.slider("Select Confidence Interval (%)", min_value=50, max_value=99, value=95)
        interval_width = confidence_interval / 100

        # Prepare data and fit model
        df = prepare_data(data, date_column, target_column)
        if df is not None:
            model = Prophet(interval_width=interval_width)
            model.fit(df)

            # Run Forecast and Hindcast
            if st.button("Run Forecast"):
                forecast = forecast_data(model, forecast_period, time_unit)
                plot_forecast(df, forecast)
                plot_forecast_components(model, forecast)  # New component trend plot

            if st.button("Run Hindcast"):
                hindcast_df, hindcast = hindcast_data(model, df, time_unit, hindcast_period)
                plot_hindcast(df, hindcast_df, hindcast)

            if st.button("Run Anomaly Detection"):
                df_with_anomalies = detect_anomalies(df.copy())
                plot_anomalies(df_with_anomalies)

# Run the app
if __name__ == "__main__":
    run_app()