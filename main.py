import json
import os
from datetime import timedelta, datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests

# Import the training module
import ml_training

contract_size = 100

def run(ticker, time_frame):
    pass
    # spot_price, option_data = scrape_data(ticker)
    # compute_total_gex(spot_price, option_data)
    # compute_gex_by_strike(spot_price, option_data)
    # compute_gex_by_expiration(option_data)
    # print_gex_surface(spot_price, option_data)
    # plot_stock_and_gamma(ticker, option_data, spot_price, time_frame)
    # plot_open_interest(option_data)

# INTRADAY ALPHA VANTAGE API (STOCK DATA)
def stock_data(ticker, time_frame):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={time_frame}&apikey=Z3HQ0DGJ9ADT15M3"
    response = requests.get(url)
    data = response.json()
    time_series_key = f"Time Series ({time_frame})"
    if time_series_key not in data:
        print(f"Error fetching stock data for {ticker}: {data.get('Note', 'No data available')}")
        return pd.DataFrame()  # Return an empty DataFrame if no data is available
    df = pd.DataFrame(data['Time Series (1min)']).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    return df


def options_data(ticker, api_key="Z3HQ0DGJ9ADT15M3"):
    url = f"https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol={ticker}&require_greeks=true&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
    
    data = response.json()
    
    if "optionChain" not in data:
        raise Exception("Invalid API response: 'optionChain' key not found")
    
    option_chain = data["optionChain"]

    # Convert to DataFrame
    df = pd.DataFrame(option_chain)
    
    # Ensure correct formatting
    if not df.empty:
        df.index = pd.to_datetime(df["expirationDate"], errors="coerce")
        df = df.drop(columns=["expirationDate"])  # Remove original date column if needed
        df = df.astype(float, errors="ignore")  # Convert numeric columns
    
    return df

def fix_option_data(data):
    """
    Fix option data columns.
    From the name of the option derive type of option, expiration and strike price.
    """
    data["type"] = data.option.str.extract(r"\d([A-Z])\d")
    data["strike"] = data.option.str.extract(r"\d[A-Z](\d+)\d\d\d").astype(int)
    data["expiration"] = data.option.str.extract(r"[A-Z](\d{6})").astype(str)
    data["expiration"] = pd.to_datetime(data["expiration"], format="%y%m%d", errors='coerce')
    return data

def compute_total_gex(data, spot_price):
    """Compute dealers' total GEX"""
    spot = spot_price
    data["GEX"] = spot * data.gamma * data.open_interest * contract_size * spot * 0.01
    data["GEX"] = data.apply(lambda x: -x.GEX if x.type == "P" else x.GEX, axis=1)
    
    total_gex = data["GEX"].sum() / 10**9
    return total_gex

def compute_gex_by_strike(spot, data):
    """Compute and plot GEX by strike using Plotly"""
    gex_by_strike = data.groupby("strike")["GEX"].sum() / 10**9
    fig = px.bar(
        x=gex_by_strike.index,
        y=gex_by_strike,
        labels={"x": "Strike Price", "y": "Gamma Exposure (Bn$)"},
        title=f"{ticker} GEX by Strike",
        color=gex_by_strike,
        color_continuous_scale="blues",
        template="plotly_dark"
    )
    # Add vertical line for spot price
    fig.add_vline(x=spot, line=dict(color="grey", dash="dash"), annotation_text="Spot Price", annotation_position="top right")
    fig.show()

def compute_gex_by_expiration(data):
    """Compute and plot GEX by expiration using Plotly"""
    selected_date = datetime.today() + timedelta(days=365)
    data = data.loc[data.expiration < selected_date]
    gex_by_expiration = data.groupby("expiration")["GEX"].sum() / 10**9
    fig = px.bar(
        x=gex_by_expiration.index,
        y=gex_by_expiration.values,
        labels={"x": "Expiration Date", "y": "Gamma Exposure (Bn$)"},
        title=f"{ticker} GEX by Expiration",
        color=gex_by_expiration.values,
        color_continuous_scale="blues",
        template="plotly_dark"
    )
    fig.update_xaxes(type="date")
    fig.show()

def print_gex_surface(spot, data):
    """Plot GEX surface using Plotly"""
    selected_date = datetime.today() + timedelta(days=365)
    limit_criteria = (
        (data.expiration < selected_date)
        & (data.strike > spot * 0.85)
        & (data.strike < spot * 1.15)
    )
    data = data.loc[limit_criteria]
    data = data.groupby(["expiration", "strike"])["GEX"].sum().unstack().fillna(0) / 10**6
    fig = go.Figure(data=[go.Surface(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale="Viridis"
    )])
    fig.update_layout(
        title="Gamma Exposure Surface",
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Expiration Date",
            zaxis_title="Gamma (M$)",
        ),
        template="plotly_dark"
    )
    fig.show()

# def fetch_stock_data(ticker, interval="1h"):
#     """Fetch historical stock price data with the given time frame (interval)"""
#     url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval={interval}&apikey=cfed9ce8186d4055b3c064a129f5e47c"
#     response = requests.get(url)
#     data = response.json()
#     if 'values' not in data:
#         raise ValueError(f"Error fetching stock data for {ticker}: {data.get('message', 'No data available')}")
    
#     df = pd.DataFrame(data['values'])
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df.set_index('datetime', inplace=True)
#     df = df.sort_index()
    
#     # Convert numeric values properly
#     for col in ['open', 'high', 'low', 'close']:
#         df[col] = df[col].astype(float)

#     return df

def plot_open_interest(data):
    """Plot open interest for calls and puts"""
    oi_by_type = data.groupby("type")["open_interest"].sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=oi_by_type.index, y=oi_by_type.values, name="Open Interest", marker_color="blue"))
    fig.update_layout(title="Open Interest by Option Type", xaxis_title="Option Type", yaxis_title="Open Interest")
    fig.show()
    
def plot_stock_and_gamma(ticker, option_data, spot_price, time_frame="1h"):
    """Plot stock price with gamma exposure markers"""
    
    # Compute moving averages for technical indicators.
    stock_data['SMA10'] = stock_data['close'].rolling(window=10).mean()
    stock_data['SMA20'] = stock_data['close'].rolling(window=20).mean()
    stock_data['SMA50'] = stock_data['close'].rolling(window=50).mean()
    stock_data['SMA100'] = stock_data['close'].rolling(window=100).mean()
    stock_data['SMA200'] = stock_data['close'].rolling(window=200).mean()

    # (Additional plotting code remains unchanged.)
    # For brevity, we assume the rest of your plotting code is here.
    print("Plotting stock and gamma exposure chart...")
    # ... (existing plotting code)
    # Example:
    # fig.show()

def scatterPlot(fig, stock_data, label):
    if label in stock_data.columns:
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data[label],
            mode='lines',
            name=label,
            line=dict(color='yellow', width=1)
        ))
    else:
        print(f"Warning: {label} not found in stock_data columns")

def prepare_data(stock_data, option_data):
    # Only create the 'date' column if it's not already present.
    if 'date' not in stock_data.columns:
        stock_data['date'] = stock_data.index.date

    # Create target: 1 if next day return > 0, else 0.
    stock_data['return'] = stock_data['close'].pct_change().shift(-1)
    stock_data['target'] = (stock_data['return'] > 0).astype(int)
    stock_data['SMA10'] = stock_data['close'].rolling(window=10).mean()
    stock_data['SMA20'] = stock_data['close'].rolling(window=20).mean()
    
    # Aggregate options data: sum GEX by day (using expiration date as a proxy).
    option_data['date'] = option_data['expiration'].dt.date
    option_features = option_data.groupby('date')['GEX'].sum().reset_index()
    
    # Merge stock and option features on date.
    data = pd.merge(stock_data, option_features, on='date', how='left')
    data['GEX'] = data['GEX'].fillna(0)
    
    return data

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

def predict(model, X_test):
    return model.predict(X_test)


def plot_stock_and_gamma_with_predictions(
    stock_data, 
    y_pred, 
    title="Stock Price with GEX and ML Predictions"
):
    """
    Plot a candlestick chart of the stock price, overlay gamma exposure (GEX),
    and show bullish/bearish predictions with up/down arrows.
    
    Parameters
    ----------
    stock_data : pd.DataFrame
        Must include columns: 'open', 'high', 'low', 'close', and optionally 'GEX'.
        The index should be a DatetimeIndex.
    y_pred : array-like
        Model predictions aligned with the rows of stock_data (1 = bullish, 0 = bearish).
    title : str
        Title of the chart.
    """

    # Make sure we have the columns we need
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in stock_data.columns:
            raise ValueError(f"stock_data is missing required column: {col}")
    
    # Create a Plotly figure
    fig = go.Figure()

    # -- 1) Candlestick Chart for Price --
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['open'],
        high=stock_data['high'],
        low=stock_data['low'],
        close=stock_data['close'],
        name='Price'
    ))

    # -- 2) Overlay GEX on a Second Y-Axis (if present) --
    if 'GEX' in stock_data.columns:
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['GEX'],
            mode='lines',
            name='GEX',
            line=dict(color='yellow', width=2),
            yaxis='y2'
        ))

    # -- 3) Plot Bullish/Bearish Predictions as Arrows --
    bullish_mask = (y_pred == 1)
    bearish_mask = (y_pred == 0)

    fig.add_trace(go.Scatter(
        x=stock_data.index[bullish_mask],
        y=stock_data['close'][bullish_mask],
        mode='markers',
        name='Bullish Prediction',
        marker=dict(symbol='triangle-up', size=12, color='green')
    ))

    fig.add_trace(go.Scatter(
        x=stock_data.index[bearish_mask],
        y=stock_data['close'][bearish_mask],
        mode='markers',
        name='Bearish Prediction',
        marker=dict(symbol='triangle-down', size=12, color='red')
    ))

    # -- 4) (Optional) Plot Moving Averages --
    # If your DataFrame has them, you can add them as extra lines:
    for ma_col in ['SMA10', 'SMA20', 'SMA50', 'SMA100', 'SMA200']:
        if ma_col in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data[ma_col],
                mode='lines',
                name=ma_col,
                line=dict(width=1)
            ))

    # -- 5) Configure Layout --
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        # Create a second y-axis for GEX on the right side
        yaxis2=dict(
            title='GEX',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )

    fig.show()


if __name__ == "__main__":
    ticker = "SPY"
    time_frame = "1h"

    # 1) Fetch data
    stock_data = stock_data(ticker, time_frame)
    option_data = options_data(ticker)

    # 2) Compute GEX

    total_gex = compute_total_gex(option_data, stock_data['close'].iloc[0])

    print(f"Total GEX for {ticker}: ${total_gex:.2f} billion")
    # 3) Prepare data for training (merges GEX into stock_data)
    data = prepare_data(stock_data, option_data)

    # 4) Train model
    model = ml_training.train_model(stock_data, option_data)

    # 5) Generate predictions
    X = data[['close', 'GEX', 'SMA10', 'SMA20']]
    y = data['target']
    y_pred = predict(model, X)

    # 6) Plot everything: price, GEX, arrows
    plot_stock_and_gamma_with_predictions(
        stock_data=data, 
        y_pred=y_pred, 
        title=f"{ticker} Price with GEX & ML Predictions"
    )