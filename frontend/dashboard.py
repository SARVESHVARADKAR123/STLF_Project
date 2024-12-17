import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

def load_data():
    """
    Load and display historical data
    """
    df = pd.read_csv('data/AAPL_processed.csv')
    return df

def plot_stock_data(df):
    """
    Create interactive plot using plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))
    
    return fig

def get_prediction(input_data):
    """
    Get prediction from backend API
    """
    try:
        response = requests.post(
            'http://localhost:5000/predict',
            json={'input_data': input_data.tolist()}
        )
        return response.json()['prediction']
    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None

def main():
    st.title('Stock Price Prediction Dashboard')
    
    # Load data
    df = load_data()
    
    # Display plot
    st.plotly_chart(plot_stock_data(df))
    
    # Prediction section
    st.subheader('Get Prediction')
    if st.button('Predict Next Day'):
        input_data = df['Close'].values[-60:].reshape(-1, 1)
        prediction = get_prediction(input_data)
        if prediction:
            st.success(f'Predicted price: ${prediction:.2f}')

if __name__ == "__main__":
    main() 