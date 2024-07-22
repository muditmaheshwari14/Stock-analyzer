import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import base64
import openai
import cohere
import plotly.graph_objects as go
import requests

# Function to encode image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Add custom CSS for background image
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        color: white; /* Text color */
    }
    /* Style for text input label */
    .stTextInput label {
        color: #FFFFFF; /* Change this to your desired label color */
        font-size: 1.2em; /* Font size for the label */
        font-weight: bold; /* Make the label bold */
    }
     .stNumberInput label {
        color: #FFFFFF; /* Change this to your desired label color */
        font-size: 1.2em; /* Font size for the label */
        font-weight: bold; /* Make the label bold */
    }
    .stTextInput > div > input {
        color: black; /* Input text color */
        background-color: white; /* Input background color */
    }
    .stNumberInput > div > input {
        color: black; /* Number input text color */
        background-color: white; /* Number input background color */
    }
    .stButton > button {
        color: white; /* Button text color */
        background-color: #007BFF; /* Button background color */
        border: none; /* Remove border */
        padding: 10px 20px; /* Add padding */
        border-radius: 5px; /* Rounded corners */
    }
    .stTextInput, .stNumberInput {
        font-size: 1.2em; /* Input font size */
        margin: 10px 0; /* Spacing around inputs */
    }
    .stButtonContainer {
        display: flex;
        justify-content: center; /* Center-align button horizontally */
        margin: 20px 0; /* Add margin around button */
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Set your background image path
set_background("app_bg1.webp")

# Define functions for creating sequences and predicting future prices
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

def predict_future_prices(model, last_sequence, n_steps):
    prediction = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        next_value = model.predict(current_sequence.reshape(1, sequence_length, 1))
        prediction.append(next_value[0, 0])
        current_sequence = np.append(current_sequence[1:], next_value, axis=0)

    return prediction

def get_quarterly_fundamentals(stock):
    try:
        ticker = yf.Ticker(stock)
        quarterly_financials = ticker.quarterly_financials
        quarterly_balance_sheet = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow
        return quarterly_financials, quarterly_balance_sheet, quarterly_cashflow
    except Exception as e:
        st.error(f"Error fetching quarterly fundamentals: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def filter_quarterly_data(df, start_date):
    # Filter columns (quarters) based on the date
    df.columns = pd.to_datetime(df.columns)
    df_filtered = df.loc[:, df.columns >= pd.to_datetime(start_date)]
    
    return df_filtered

def get_market_price(stock):
    stock = yf.Ticker(stock)
    return stock.history(period='1d')['Close'].iloc[-1]

def calculate_pe_ratios(market_price, eps_series):
    return [market_price / eps if eps > 0 else None for eps in eps_series]


def fetch_news(stock, api_key):
    url = f"https://newsapi.org/v2/everything?q={stock}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    return news_data.get('articles', [])

# Streamlit header
st.markdown('<p style="color: white; font-size: 48px; text-align: center;">Stock Price Predictor</p>', unsafe_allow_html=True)

stock = st.text_input(label='Enter Stock Symbol (e.g., RELIANCE.NS)',label_visibility='visible',value='GOOG')

weeks = st.number_input( label='Enter no. of weeks for prediction',label_visibility='visible',min_value=1, max_value=52)

news_api_key = "455d45b1175146fe88765ea9d505ba9a"

# Stock input with submit button
        
if st.button('Submit'):
    try:
            start = '2012-01-01'
            data = yf.download(stock, start)

            if data.empty:
                st.markdown('<p style="color: white; font-size: 36px; text-align: center;">Invalid stock symbol. Please enter a valid stock symbol.</p>', unsafe_allow_html=True)
                st.error(' ')
            else:
                 start = '2012-01-01'

            quarterly_financials, quarterly_balance_sheet, quarterly_cashflow = get_quarterly_fundamentals(stock)
            filtered_financials = filter_quarterly_data(quarterly_financials, start)

            essential_rows = [
    'Total Revenue', 'Gross Profit', 'Operating Income', 'EBITDA', 'Net Income',
    'Diluted EPS', 'Cost Of Revenue', 'Operating Expense'
    
]
            df = filtered_financials
            df_filter = df.loc[df.index.isin(essential_rows)]
            market_price = get_market_price(stock)
            eps_values = df_filter.loc['Diluted EPS'] if 'Diluted EPS' in df_filter.index else pd.Series()
            pe_ratios = calculate_pe_ratios(market_price, eps_values)
            if not eps_values.empty:
                df_filter.loc['P/E Ratio'] = pe_ratios

            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Stock Data</p>', unsafe_allow_html=True)
            st.write(data)

            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Price vs MA50</p>', unsafe_allow_html=True)
            ma_50_days = data.Close.rolling(50).mean()
            ma_100_days = data.Close.rolling(100).mean()
            fig1 = plt.figure(figsize=(8,6))
            plt.plot(ma_50_days, 'r', label='50-Day Moving Average')
            plt.plot(ma_100_days, 'b', label='100-Day Moving Average')
            plt.plot(data.Close, 'g', label='Close Price')
            plt.legend()
            plt.show()
            st.pyplot(fig1)

            

    # Prepare data for model
            data_close = data[['Close']]
            scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_close.values)

            sequence_length = 60
            X, y = create_sequences(scaled_data, sequence_length)

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

    # Reshape for LSTM (samples, time steps, features)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Load pre-trained model
            model = tf.keras.models.load_model("C:\\Users\\Mudit\\AppData\\Local\\Programs\\Python\\Python312\\Scripts\\stock_basic_app\\prediction1_model.keras")

    # Predict test data
            pred = model.predict(X_test)
            y_test_inverse = MinMaxScaler().fit(data_close.values).inverse_transform(y_test.reshape(-1, 1))
            y_pred_inverse = MinMaxScaler().fit(data_close.values).inverse_transform(pred)

            test_dates = data_close.index[-len(y_test_inverse):]

    # Plot the results
            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Original Price Vs Predicted Price</p>', unsafe_allow_html=True)
            fig2 = plt.figure(figsize=(14,10))
            plt.plot(test_dates, y_test_inverse, color='blue', label='Actual Stock Price')
            plt.plot(test_dates, y_pred_inverse, color='red', label='Predicted Stock Price')
            plt.title('Stock Price Prediction')
            plt.xlabel('Year')
            plt.ylabel('Stock Price')
            plt.legend()
            plt.show()
            st.pyplot(fig2)

    # Predict future stock prices for user-defined weeks
            last_sequence = scaled_data[-sequence_length:]
            future_predictions = predict_future_prices(model, last_sequence, weeks * 5)  # Assuming 5 trading days per week
            future_predictions = MinMaxScaler().fit(data_close.values).inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
            future_dates = pd.date_range(start=data_close.index[-1], periods=weeks * 5 + 1, freq='B')[1:]  # 'B' for business days

            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Future Stock Price Predictions</p>', unsafe_allow_html=True)
            future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close Price'])
            st.write(future_df)

        # Plot future predictions
            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Trend for Future Predictions</p>', unsafe_allow_html=True)
            fig3 = plt.figure(figsize=(14,10))
            plt.plot(future_dates, future_predictions, color='purple', label='Future Predicted Stock Price')
            plt.title('Future Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Predicted Stock Price')
            plt.legend()
            plt.show()
            st.pyplot(fig3)

            cohere_client = cohere.Client('noofMGWfzrLf5Y9lFUXN9fIX3ecpCoW1aRh1UcRj')

            

            metrics = {
        'EBITDA': df_filter.loc['EBITDA'].mean() if 'EBITDA' in df_filter.index else np.nan,
        'Diluted EPS': df_filter.loc['Diluted EPS'].mean() if 'Diluted EPS' in df_filter.index else np.nan,
        'Net Income': df_filter.loc['Net Income'].mean() if 'Net Income' in df_filter.index else np.nan,
        'Operating Income': df_filter.loc['Operating Income'].mean() if 'Operating Income' in df_filter.index else np.nan,
        'Operating Expense': df_filter.loc['Operating Expense'].mean() if 'Operating Expense' in df_filter.index else np.nan,
        'Gross Profit': df_filter.loc['Gross Profit'].mean() if 'Gross Profit' in df_filter.index else np.nan,
        'Cost Of Revenue': df_filter.loc['Cost Of Revenue'].mean() if 'Cost Of Revenue' in df_filter.index else np.nan,
        'Total Revenue': df_filter.loc['Total Revenue'].mean() if 'Total Revenue' in df_filter.index else np.nan,
        'P/E Ratio': df_filter.loc['P/E Ratio'].mean() if 'P/E Ratio' in df_filter.index else np.nan
    }
            available_metrics = {metric: value for metric, value in metrics.items() if not np.isnan(value)}

            def calculate_rating(metrics):
                positive_factors = [
                    'EBITDA',
                    'Net Income',
                    'Operating Income',
                    'Gross Profit',
                    'Total Revenue'
                ]
    
                negative_factors = [
                    'Operating Expense',
                    'Cost Of Revenue'
                    'P/E Ratio'
                ]
    
                positive_score = sum([metrics[factor] for factor in positive_factors if factor in metrics])
                negative_score = sum([metrics[factor] for factor in negative_factors if factor in metrics])
    
                rating = 10 * (positive_score - negative_score) / (positive_score + negative_score)
                rating = max(1, min(10, rating))
                return rating

            rating = calculate_rating(metrics)

# Prepare the input text for the language model
            input_text = f"""
            Stock Analysis Report:
            -----------------------
            {', '.join([f"{metric}: {value:.2f}" for metric, value in available_metrics.items()])}
            Rating: {rating:.2f}/10

            Based on the above metrics, provide a detailed analysis of the stock's performance and strength and future movement.
            """

    # Generate the report using Cohere's language model
            response = cohere_client.generate(
                model='command-xlarge-nightly',
                prompt=input_text,
                max_tokens=1000
            )

            if hasattr(response, 'generations'):
    # Accessing the text from the first generation
                formatted_response = response.generations[0].text
            else:
    # Directly accessing the text if it's a single Generation object
                formatted_response = response.text

# Display the formatted response in the sidebar
            header_text = f"<h2 style='color: #FF5733;font-size: 48px;'>{'Fundamental Report of ' + stock}</h2>"
            formatted_response = f"<<h2 style='color: #000000;'>{formatted_response}</h2>"
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rating,
                title={'text': "Stock Rating", 'font': {'size': 24, 'color': "darkblue"}},
                gauge={
                    'axis': {'range': [1, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [1, 3], 'color': "red"},
                        {'range': [3, 6], 'color': "yellow"},
                        {'range': [6, 10], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "purple", 'width': 4},
                        'thickness': 0.75,
                        'value': rating
                    }
                }
            ))

            fig.update_layout(
                paper_bgcolor="lavender",
                font={'color': "darkblue", 'family': "Arial"}
            )

# Display the styled header and formatted response in the sidebar
            st.sidebar.markdown(header_text, unsafe_allow_html=True)
            st.sidebar.markdown(formatted_response, unsafe_allow_html=True,)
            st.sidebar.plotly_chart(fig)

            news_articles = fetch_news(stock, news_api_key)

    # Display recent news
            st.sidebar.markdown('<p style="color: black; font-size: 2.5em; text-align: center;">News</p>', unsafe_allow_html=True)
            
            if isinstance(news_articles, list) and len(news_articles) > 0:
                for article in news_articles[:3]:  # Display top 5 articles
                    if isinstance(article, dict):  # Ensure article is a dictionary
                        st.sidebar.markdown(f"### {article.get('title', 'No title')}")
                        st.sidebar.markdown(f"<p style='color: black;'>Published at: {article.get('publishedAt', 'No date')}</p>", unsafe_allow_html=True)
                        st.sidebar.markdown(f"[Read more]({article.get('url', '#')})")
                        st.sidebar.markdown("---")
                    else:
                        st.sidebar.markdown("Invalid article format.")
            else:
                st.sidebar.markdown('<p style="color: black; font-size: 1.5em; text-align: center;">No Recent News</p>', unsafe_allow_html=True)

    except Exception as e:
            st.error(" ")
