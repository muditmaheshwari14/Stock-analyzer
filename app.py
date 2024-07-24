import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import base64
import cohere
import plotly.graph_objects as go
import requests
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

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
        background-color:black;
        color: white; /* Text color */
    }
    .stTextInput label {
        color: #FFFFFF;
        font-size: 1.2em;
        font-weight: bold;
    }
    .stNumberInput label {
        color: #FFFFFF;
        font-size: 1.2em;
        font-weight: bold;
    }
    .stTextInput > div > input {
        color: black;
        background-color: white;
    }
    .stNumberInput > div > input {
        color: black;
        background-color: white;
    }
    .stButton > button {
        color: white;
        background-color: #007BFF;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButtonContainer {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

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

@st.cache_data
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
    df.columns = pd.to_datetime(df.columns)
    df_filtered = df.loc[:, df.columns >= pd.to_datetime(start_date)]
    return df_filtered

@st.cache_data
def get_market_price(stock):
    stock = yf.Ticker(stock)
    return stock.history(period='1d')['Close'].iloc[-1]

def calculate_pe_ratios(market_price, eps_series):
    return [market_price / eps if eps > 0 else None for eps in eps_series]

@st.cache_data
def fetch_news(stock, api_key, num_articles=5):
    url = f"https://newsapi.org/v2/everything?q={stock}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    return news_data.get('articles', [])

# Load pre-trained model
model_path = "prediction1_model.keras"
model = tf.keras.models.load_model(model_path)

# Streamlit sidebar with option menu
with st.sidebar:
    selected_tab = option_menu(
        menu_title="Navigation",
        options=["Home", "Predict Price", "Fundamental Analysis", "Technical Analysis", "News"],
        icons=["house", "graph-up-arrow", "file-earmark-text", "bar-chart", "newspaper"],
        menu_icon="cast",
        default_index=0,
    )

# Home tab
if selected_tab == "Home":
    st.markdown('<p style="color: white; font-size: 48px; text-align: center;">Stock Price Predictor</p>', unsafe_allow_html=True)

    stock = st.text_input(label='Enter Stock Symbol (e.g., GOOG)', label_visibility='visible')
    weeks = st.number_input(label='Enter number of weeks for prediction', label_visibility='visible', min_value=1, max_value=52)

    if st.button('Submit'):
        st.session_state.stock = stock
        st.session_state.weeks = weeks

# Predict Price tab
elif selected_tab == "Predict Price":
    if 'stock' in st.session_state and 'weeks' in st.session_state:
        stock = st.session_state.stock
        weeks = st.session_state.weeks
        data = yf.download(stock, start='2012-01-01',progress=False)

        if not data.empty:
            data_close = data[['Close']]
            scaled_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_close.values)

            sequence_length = 60
            X, y = create_sequences(scaled_data, sequence_length)

            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            last_sequence = scaled_data[-sequence_length:]
            future_predictions = predict_future_prices(model, last_sequence, weeks * 5)  # 5 trading days per week
            future_predictions = MinMaxScaler().fit(data_close.values).inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
            future_dates = pd.date_range(start=data_close.index[-1], periods=weeks * 5 + 1, freq='B')[1:]

            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Future Stock Price Predictions</p>', unsafe_allow_html=True)
            future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close Price'])
            st.write(future_df)

            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Trend for Future Predictions</p>', unsafe_allow_html=True)
            fig3 = plt.figure(figsize=(14, 10))
            plt.plot(future_dates, future_predictions, color='purple', label='Future Predicted Stock Price')
            plt.title('Future Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Predicted Stock Price')
            plt.legend()
            st.pyplot(fig3)
        else:
            st.error("No data found for the selected stock.")

# Fundamental Analysis tab
elif selected_tab == "Fundamental Analysis":
    if 'stock' in st.session_state:
        stock = st.session_state.stock
        st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Fundamental Analysis Report</p>', unsafe_allow_html=True)
        
        quarterly_financials, quarterly_balance_sheet, quarterly_cashflow = get_quarterly_fundamentals(stock)
        filtered_financials = filter_quarterly_data(quarterly_financials, '2012-01-01')
        
        essential_rows = [
            'Total Revenue', 'Gross Profit', 'Operating Income', 'EBITDA', 'Net Income',
            'Diluted EPS', 'Cost Of Revenue', 'Operating Expense'
        ]
        
        df_filter = filtered_financials.loc[filtered_financials.index.isin(essential_rows)]
        market_price = get_market_price(stock)
        eps_values = df_filter.loc['Diluted EPS'] if 'Diluted EPS' in df_filter.index else pd.Series()
        pe_ratios = calculate_pe_ratios(market_price, eps_values)
        if not eps_values.empty:
            df_filter.loc['P/E Ratio'] = pe_ratios

        # Calculate the rating
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
                'Cost Of Revenue',
                'P/E Ratio'
            ]
    
            positive_score = sum([metrics[factor] for factor in positive_factors if factor in metrics])
            negative_score = sum([metrics[factor] for factor in negative_factors if factor in metrics])
    
            rating = 10 * (positive_score - negative_score) / (positive_score + negative_score)
            rating = max(1, min(10, rating))
            return rating

        rating = calculate_rating(metrics)
        
        # Prepare the input text for Cohere's language model
        input_text = f"""
        Stock Analysis Report:
        -----------------------
        {', '.join([f"{metric}: {value:.2f}" for metric, value in available_metrics.items()])}
        Rating: {rating:.2f}/10

        Based on the above metrics, provide a detailed analysis of the stock's performance and strength and future movement.
        """

        # Generate the report using Cohere's language model
        cohere_client = cohere.Client('noofMGWfzrLf5Y9lFUXN9fIX3ecpCoW1aRh1UcRj')  # Replace with your Cohere API key
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
        
        # Display the formatted response in the main section
        st.write(f"**Fundamental Report for {stock}:**")
        st.write(formatted_response)

        # Display the rating gauge
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

        st.plotly_chart(fig)
    else:
        st.write("Please enter a valid stock ticker symbol.")


# Technical Analysis tab
elif selected_tab == "Technical Analysis":
    if 'stock' in st.session_state:
        stock = st.session_state.stock
        data = yf.download(stock, start='2012-01-01')

        if not data.empty:
            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Stock Price and Volume Chart</p>', unsafe_allow_html=True)
            
            fig1 = plt.figure(figsize=(14, 10))
            plt.plot(data.index, data['Close'], color='cyan', label='Stock Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Stock Price Chart for {stock}')
            plt.legend()
            st.pyplot(fig1)

            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Volume Chart</p>', unsafe_allow_html=True)
            fig2 = plt.figure(figsize=(14, 10))
            plt.bar(data.index, data['Volume'], color='red', label='Volume')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.title(f'Volume Chart for {stock}')
            plt.legend()
            st.pyplot(fig2)

            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">General Trend of the Stock</p>', unsafe_allow_html=True)
            trend = np.polyfit(range(len(data)), data['Close'], 1)
            trend_line = np.polyval(trend, range(len(data)))
            fig4 = plt.figure(figsize=(14, 10))
            plt.plot(data.index, data['Close'], label='Stock Price')
            plt.plot(data.index, trend_line, label='Trend Line', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Trend Analysis for {stock}')
            plt.legend()
            st.pyplot(fig4)
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Plot moving averages
            st.markdown('<p style="color: white; font-size: 1.5em; text-align: center;">Moving Average</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))
            fig.update_layout(title='Stock Price and Moving Averages', xaxis_title='Date', yaxis_title='Price')
            
            st.plotly_chart(fig)
        else:
            st.error("No data found for the selected stock.")
    else:
        st.warning('Please enter a stock symbol in the Home tab.')

# News tab
elif selected_tab == "News":
    if 'stock' in st.session_state:
        stock = st.session_state.stock
        api_key = 'fdc82be448d94b199b3b54ae268f4b1d'  # Replace with your News API key
        news_articles = fetch_news(stock, api_key)
        
        st.markdown(f'<p style="color: white; font-size: 1.5em; text-align: center;">Latest News about {stock}</p>', unsafe_allow_html=True)

        for article in news_articles:
            st.markdown(f"<h3 style='color:white;'>{article['title']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:gray;'>{article['description']}</p>", unsafe_allow_html=True)
            st.markdown(f"<a style='color:white;' href='{article['url']}'>Read more</a>", unsafe_allow_html=True)
    else:
        st.warning('Please enter a stock symbol in the Home tab.')


