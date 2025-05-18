import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.stats import norm
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import plotly.express as px
import os
from streamlit_autorefresh import st_autorefresh
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from functools import wraps
from yfinance.exceptions import YFRateLimitError
import time


genai.configure(api_key="AIzaSyBdT5eA5-U5jveNQTkDZkpeTpdEee_ArRA") 

st.set_page_config(page_title="Aplikasi Trading & Investasi Saham", layout="wide")

custom_css = """
<style>
body { background-color: #f4f6f9; }
h1, h2, h3, h4, h5, h6 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; }
div[data-testid="stAppViewContainer"] { padding: 2rem; }
.st-radio label { font-size: 16px; font-weight: 600; }
.stNumberInput > div { width: 100%; }
.buy-button, .sell-button {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    color: white;
    margin-top: 10px;
}
.buy-button { background-color: #28a745; }
.buy-button:hover { background-color: #218838; }
.sell-button { background-color: #dc3545; }
.sell-button:hover { background-color: #c82333; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

responsive_css = """
<style>
@media only screen and (max-width: 600px) {
  div[data-testid="stAppViewContainer"] { padding: 1rem !important; }
  .css-1d391kg { display: none !important; }
  .css-1e5imcs, .css-1kyxreq { flex-direction: column !important; width: 100% !important; }
  .buy-button, .sell-button { width: 100% !important; }
}
</style>
"""
st.markdown(responsive_css, unsafe_allow_html=True)

st.title("📊 AFA Stock Trading")

if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {"cash": 10_000_000_000.0, "positions": {}}

@st.cache_data(ttl=300)
def get_fx_rate_to_idr(currency: str) -> float:
    if currency == 'IDR': return 1.0
    pair = f"{currency}IDR=X"
    fx = yf.Ticker(pair)
    hist = fx.history(period="1d")
    return hist['Close'].iloc[-1] if not hist.empty else None

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="30d")
    info = getattr(stock, 'info', {})
    if hist.empty: raise ValueError(f"No data for '{ticker}'")
    price = hist['Close'].iloc[-1]
    return price, hist, info

st_autorefresh(interval=300000, limit=1000, key="ticker_autorefresh")

def get_stock_data_bs(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="30d")
    try:
        info = stock.info
    except:
        info = {}
    if hist.empty:
        raise ValueError(f"Tidak ada data untuk ticker '{ticker}'.")
    
    S = hist['Close'].iloc[-1]
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    sigma = np.std(returns) * np.sqrt(252)
    return S, sigma, hist, info

def create_dataset(data, time_step=90):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step, 0])
        Y.append(data[i+time_step, 0])
    return np.array(X), np.array(Y)

def run_lstm_model(ticker, time_step=90, epochs=100, batch_size=32, model_path="lstm_model.h5"):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3285)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        st.error(f"Tidak ada data untuk ticker '{ticker}'.")
        return None, None, None, None, None 
    
    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    X, Y = create_dataset(scaled_data, time_step)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    Y_train, Y_test = Y[:train_size], Y[train_size:]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        with st.spinner("Melatih model LSTM..."):
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                      epochs=epochs, batch_size=batch_size, verbose=0)
        model.save(model_path)
        st.success("✅ Model berhasil dilatih dan disimpan.")

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_train_actual = scaler.inverse_transform(Y_train.reshape(-1, 1))
    Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
    
    last_60_days = scaled_data[-time_step:]
    next_input = last_60_days.reshape((1, time_step, 1))
    next_day_pred = model.predict(next_input)
    next_day_pred_price = scaler.inverse_transform(next_day_pred)[0][0]

    test_index = data.index[-len(Y_test_actual):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_index, y=Y_test_actual.flatten(), mode='lines', name='Aktual'))
    fig.add_trace(go.Scatter(x=test_index, y=test_predict.flatten(), mode='lines', name='Prediksi'))
    fig.update_layout(title=f'📈 Prediksi Harga Saham {ticker} Menggunakan LSTM',
                  xaxis_title='Tanggal', yaxis_title='Harga Saham')

    st.session_state['lstm_fig'] = fig
    st.plotly_chart(fig, use_container_width=True)

    
    return sqrt(mean_squared_error(Y_train_actual, train_predict)), \
           sqrt(mean_squared_error(Y_test_actual, test_predict)), \
           {
               "Train": Y_train_actual.flatten(),
               "Valid": Y_test_actual.flatten(),
               "Prediction": test_predict.flatten()
           }, model, next_day_pred_price

def detect_trend(data, window=10):
    close_prices = data['Close'][-window:]
    start_price = close_prices.iloc[0]
    end_price = close_prices.iloc[-1]
    percent_change = ((end_price - start_price) / start_price) * 100

    if percent_change > 2:
        return f"📈Uptrend (+{percent_change:.2f}%)"
    elif percent_change < -2:
        return f"📉Downtrend ({percent_change:.2f}%)"
    else:
        return f"➡️Sideways ({percent_change:.2f}%)"

st.sidebar.markdown("## 🧭 Navigasi")
ticker = st.sidebar.text_input("Masukkan Ticker Saham", "AAPL", help="Contoh: AAPL, TSLA, BBRI.JK")

markets = {
    "Amerika": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "NVDA", "BRK-B", "JPM"],
    "Eropa": ["BMW.DE", "ADS.DE", "AIR.PA", "BN.PA", "SAP.DE", "VOW3.DE", "ALV.DE", "DTE.DE"],
    "Asia": ["7203.T", "6758.T", "9984.T", "TCS.NS", "005930.KS", "005380.KS", "INFY.NS", "9988.HK", "2330.TW"]
}

for region, tickers in markets.items():
    with st.sidebar.expander(region):
        for t in tickers:
            with st.container():
                try:
                    stock = yf.Ticker(t)
                    info = stock.info
                    previous_close = info.get("previousClose")
                    hist = stock.history(period="1d", interval="5m")
                    
                    if previous_close and not hist.empty:
                        current_price = hist["Close"].iloc[-1]
                        growth = ((current_price - previous_close) / previous_close) * 100
                        growth_str = f"{growth:.2f}%"
                        chart_color = "royalblue" if growth >= 0 else "red"
                    else:
                        growth_str = "N/A"
                        chart_color = "gray"
                    
                    fig = go.Figure(data=go.Scatter(
                        x=hist.index,
                        y=hist["Close"],
                        mode="lines",
                        line=dict(width=1, color=chart_color)
                    ))
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=60,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False)
                    )
                    
                    currency = info.get('currency', 'USD')
                    currency_symbols = {
                        'USD': '$','IDR': 'Rp','EUR': '€','JPY': '¥','GBP': '£',
                        'AUD': 'A$','CAD': 'C$','CHF': 'CHF','CNY': '¥','HKD': 'HK$',
                        'SGD': 'S$','KRW': '₩','INR': '₹','BRL': 'R$','ZAR': 'R','MXN': '$',
                        'RUB': '₽','TRY': '₺','SEK': 'kr','NOK': 'kr','DKK': 'kr',
                        'PLN': 'zł','TWD': 'NT$','THB': '฿','MYR': 'RM',  
                    }
                    symbol = currency_symbols.get(currency, currency + ' ')
                    
                    col_info, col_chart = st.columns([1, 2])
                    with col_info:
                        st.markdown(f"**{t}**")
                        st.markdown(f"*{growth_str}*")
                        st.markdown(f"{symbol}{current_price:.2f}")
                    with col_chart:
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.markdown(f"**{t}**")
                    st.markdown("N/A")
                    st.error(f"Error: {e}")
                    
                st.markdown("---")
                
def get_financials(ticker):
    stock = yf.Ticker(ticker)
    fin = stock.financials.T if hasattr(stock, "financials") else pd.DataFrame()
    bs  = stock.balance_sheet.T if hasattr(stock, "balance_sheet") else pd.DataFrame()
    cf  = stock.cashflow.T if hasattr(stock, "cashflow") else pd.DataFrame()
    return fin, bs, cf

def get_analyst_info(ticker):
    stock = yf.Ticker(ticker)
    rec = stock.recommendations if hasattr(stock, "recommendations") else pd.DataFrame()
    earnings = stock.earnings if hasattr(stock, "earnings") else pd.DataFrame()
    return rec, earnings

def get_major_holders(ticker):
    stock = yf.Ticker(ticker)
    maj = stock.major_holders if hasattr(stock, "major_holders") else None
    inst = stock.institutional_holders if hasattr(stock, "institutional_holders") else None
    return maj, inst

def get_ai_analysis(prompt: str) -> str:
    """Panggil Google Generative AI untuk analisis AFA."""
    model = genai.GenerativeModel(
        model_name='models/gemini-2.0-flash',
        system_instruction=(
            "Anda adalah asisten ahli finansial yang akan membuat "
            "analisis AFA (Analisis Fundamental & Analisis Teknikal)."
        )
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=2048
        ),
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT:    HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH:   HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }
    )
    return response.text

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Jual & Beli saham",
    "🎯 Analisis Saham",
    "💬 AI Chat Finansial & Investasi",
    "📒 Portofolio Saya"
])

with tab1: 
    st.header("📊 Jual & Beli saham")
    if "prev_ticker" in st.session_state:
        if st.session_state["prev_ticker"] != ticker:
            st.session_state.pop("lstm_result", None)
    st.session_state["prev_ticker"] = ticker

    try:
        S2, sigma2, hist2, info2 = get_stock_data_bs(ticker)
    except Exception as e:
        st.error("Gagal mengambil data saham: " + str(e))
        S2, sigma2, hist2, info2 = None, None, None, None
       
     
    currency = info2.get('currency', 'USD') if info2 else 'USD'
    currency_symbols = {
        'USD': '$','IDR': 'Rp','EUR': '€','JPY': '¥','GBP': '£',
        'AUD': 'A$','CAD': 'C$','CHF': 'CHF','CNY': '¥','HKD': 'HK$',
        'SGD': 'S$','KRW': '₩','INR': '₹','BRL': 'R$','ZAR': 'R','MXN': '$',
        'RUB': '₽','TRY': '₺','SEK': 'kr','NOK': 'kr','DKK': 'kr',
        'PLN': 'zł','TWD': 'NT$','THB': '฿','MYR': 'RM',  
    }
    symbol = currency_symbols.get(currency, currency + ' ')
    
    period = st.selectbox("Pilih periode data:", ["1d","7d","30d","1mo"])
    
    try:
        intraday = yf.download(ticker, period = period, interval="5m")
        if not intraday.empty:
            last_price = float(intraday['Close'].iloc[-1])

            y = intraday['Close'].values.flatten()
            x = intraday.index

            fig = go.Figure(go.Scatter(
                x=x, y=y, mode='lines', name='Intraday 5m'
            ))
            fig.update_layout(
                title="Grafik Harga Intraday (5 Menit)",
                xaxis_title="Waktu",
                yaxis_title="Harga Penutupan",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Data intraday tidak tersedia untuk ticker ini.")
    except Exception as e:
        st.error(f"Gagal mengambil data intraday: {e}")
    
    try:
        price, hist, info = get_stock_data(ticker)
    except Exception as e:
        st.error(f"Gagal ambil data: {e}")
        st.stop()
    currency = info.get('currency','USD')
    fx = get_fx_rate_to_idr(currency) or 1.0
    price_idr = price * fx
    
    with st.form("form_trading"):
        col1,col2 = st.columns(2)
        with col1:
            qty = st.number_input("Jumlah Saham", min_value=1, value=1, step=1)
            total = qty * price_idr
            kolom1, kolom2, kolom3, kolom4, kolom5 = st.columns(5)
            with kolom1:
                beli = st.form_submit_button("🟢 Beli")
            with kolom2:
                jual = st.form_submit_button("🔴 Jual")
            with kolom3:
                st.markdown("         ")
            with kolom4:
                st.markdown("         ")
            
            try:
                last_price = float(hist2['Close'].iloc[-1])
            except:
                last_price = 0.0
            total_cost= qty * price_idr
        
        with col2:
            st.metric("Harga per Saham (IDR)", f"Rp{price_idr:,.2f}")
            st.metric("Total (IDR)", f"Rp{total:,.2f}")
        
        if beli or jual:
            action = "Beli" if beli else "Jual"
            port = st.session_state.get("portfolio", {"cash":0, "positions":{}})
            if action == "Beli":
                if port["cash"] >= total_cost:
                    port["cash"] -= total_cost
                    port["positions"][ticker] = port["positions"].get(ticker, 0) + qty
                    st.success(f"✅ Berhasil membeli {qty} saham {ticker} dengan harga {symbol}{last_price:,.2f}")
                else:
                    st.error("❌ Saldo kas tidak mencukupi.")
            else:
                held = port["positions"].get(ticker, 0)
                if held >= qty:
                    port["positions"][ticker] -= qty
                    port["cash"] += total_cost
                    st.success(f"✅ Berhasil menjual {qty} saham {ticker} dengan seharga RP{total_cost:,.2f}")
                else:
                    st.error("❌ Jumlah saham yang dimiliki tidak mencukupi.")
            st.session_state["portfolio"] = port
            
    st.markdown("---")
    stock = yf.Ticker(ticker)
    info = stock.info
    fin, bs, cf = get_financials(ticker)
    
    targetHighPrice = info.get("targetHighPrice")
    
    st.subheader("🏢 Profil Perusahaan")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Nama:** {info.get('longName', '-')}") 
        st.markdown(f"**Ticker:** {ticker}")
        st.markdown(f"**Sektor:** {info.get('sector', '-')}")
        st.markdown(f"**Industri:** {info.get('industry', '-')}")
        st.markdown(f"**Negara:** {info.get('country', '-')}")
    with col2:
        market_cap = info.get("marketCap")
        fwd_rate  = info.get("forwardAnnualDividendRate")
        fwd_yield = info.get("forwardAnnualDividendYield")
        st.metric("Market Cap", f"{symbol}{market_cap:,}" if market_cap else "-")
        st.metric("EPS (TTM)", info.get("trailingEps", "-"))
        st.metric("Average Volume ", info.get("averageVolume"))
        st.metric("Target 1 tahun estimasi",f"{symbol}{targetHighPrice:.2f}")
    
    st.subheader("⚙️ Rasio Fundamental")
    ratios = {
        'Trailing P/E': info.get('trailingPE'),
        'Forward P/E': info.get('forwardPE'),
        'Price/Book': info.get('priceToBook'),
        'Price/Sales': info.get('priceToSalesTrailing12Months'),
        'Beta': info.get('beta'),
        'Return of equity': info.get('returnOnEquity'),
        'Return of assets': info.get('returnOnAssets'),
        'Dividend Yield': info.get('dividendYield'),
        'Current Ratio': info.get('currentRatio'),
        'Quick Ratio': info.get('quickRatio')
    }
    st.table(pd.DataFrame.from_dict(ratios, orient='index', columns=['Value']))
    
    fin, bs, cf = get_financials(ticker)
    with st.expander("Laporan Laba Rugi (Income Statement)"):
        st.dataframe(fin)
    with st.expander("Neraca (Balance Sheet)"):
        st.dataframe(bs)
    with st.expander("Arus Kas (Cash Flow)"):
        st.dataframe(cf)
    
    rec, earnings = get_analyst_info(ticker)
    with st.expander("Rekomendasi Analis"):
        if rec is not None and not rec.empty:
            st.dataframe(rec.tail(10))
        else:
            st.write("Data rekomendasi tidak tersedia.")
    
    maj, inst = get_major_holders(ticker)
    with st.expander("Major Holders"):
        st.write(maj)
        st.write(inst)
        
    st.subheader("🗓️ Kalender Keuangan")
    cal = stock.calendar  

    try:
        df_cal = pd.DataFrame.from_dict(cal, orient='index', columns=['Value'])
    except Exception:
        df_cal = pd.DataFrame()

    if not df_cal.empty:
        st.table(df_cal)
    else:
        st.write("— Tidak ada data kalender —")
    
with tab2: 
    st.subheader("📈 Prediksi Harga Saham dengan LSTM")
    
    if "prev_ticker" in st.session_state:
        if st.session_state["prev_ticker"] != ticker:
            st.session_state.pop("lstm_result", None)
    st.session_state["prev_ticker"] = ticker

    try:
        S2, sigma2, hist2, info2 = get_stock_data_bs(ticker)
    except Exception as e:
        st.error("Gagal mengambil data saham: " + str(e))
        S2, sigma2, hist2, info2 = None, None, None, None

    currency = info2.get('currency', 'USD') if info2 else 'USD'
    currency_symbols = {
                        'USD': '$','IDR': 'Rp','EUR': '€','JPY': '¥','GBP': '£',
                        'AUD': 'A$','CAD': 'C$','CHF': 'CHF','CNY': '¥','HKD': 'HK$',
                        'SGD': 'S$','KRW': '₩','INR': '₹','BRL': 'R$','ZAR': 'R','MXN': '$',
                        'RUB': '₽','TRY': '₺','SEK': 'kr','NOK': 'kr','DKK': 'kr',
                        'PLN': 'zł','TWD': 'NT$','THB': '฿','MYR': 'RM',  
                    }
    
    symbol = currency_symbols.get(currency, currency + ' ')

    if 'lstm_result' in st.session_state:
        result = st.session_state['lstm_result']
        trend = detect_trend(hist2) if hist2 is not None else "N/A"
        if 'lstm_fig' in st.session_state:
            st.plotly_chart(st.session_state['lstm_fig'], use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        cols = st.columns(2)
        cols[0].metric("Train RMSE", f"{result['train_rmse']:.2f}")
        cols[1].metric("Test RMSE", f"{result['test_rmse']:.2f}")
        cols = st.columns(2)
        cols[0].metric("Prediksi Harga Besok", f"{symbol}{result['next_day_price']:.2f}")
        cols[1].metric("Harga Saham saat ini", f"{symbol}{S2:,.2f}" if S2 is not None else "N/A")
        
    else: 
        with st.spinner("⏳ Melatih model dan memprediksi harga..."):
                train_rmse, test_rmse, lstm_history, model, next_day_price = run_lstm_model(ticker)
                st.session_state['lstm_result'] = {
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "lstm_history": lstm_history,
                    "model": model,
                    "next_day_price": next_day_price
                }
                trend = detect_trend(hist2) if hist2 is not None else "N/A"
                col1, col2, col3, col4 = st.columns(4)
                cols = st.columns(2)
                cols[0].metric("Train RMSE", f"{train_rmse:.2f}")
                cols[1].metric("Test RMSE", f"{test_rmse:.2f}")
                cols = st.columns(2)
                cols[0].metric("Prediksi Harga Besok", f"{symbol}{next_day_price:.2f}")
                cols[1].metric("Harga Saham saat ini", f"{symbol}{S2:,.2f}" if S2 is not None else "N/A")
    st.markdown("---")
    
    st.subheader("💡 Analisis AFA")
    info2 = {
        "marketCap": "2.5T",
        "trailingPE": 28.5,
        "dividendYield": "0.55%"
    }
    signals = [
        "SMA50 > SMA200 menunjukkan tren naik",
        "RSI 14 hari: 65 (mendekati area jenuh beli)",
        "MACD crossover: bullish",
    ]
    
    signals_str = "\n".join(f"   - {sig}" for sig in signals)
    fundamental = {
        "Market Cap": info2.get("marketCap"),
        "Trailing P/E": info2.get("trailingPE"),
        "Dividend Yield": info2.get("dividendYield")
    }
    
    deep_prompt = f"""
    Anda adalah seorang Analis Keuangan Senior dengan pengalaman lebih dari 15 tahun, bersertifikat CFA dan ahli di pasar modal. Tugas Anda adalah menyusun Analisis AFA (Analisis Fundamental & Analisis Teknikal) yang sangat mendalam untuk saham {ticker}.

    1. Konteks Makro dan Sektor
    - Gambarkan kondisi ekonomi global dan domestik terkini (inflasi, suku bunga, pertumbuhan GDP, kebijakan moneter)
    - Jelaskan tren utama di sektor industri tempat {ticker} beroperasi (pertumbuhan, ancaman, peluang regulasi)

    2. Analisis Fundamental
    - Valuasi: Bandingkan rasio P/E, P/BV, EV/EBITDA {ticker} vs. rata-rata sektor dan pesaing utama
    - Profitabilitas: Uraikan margin kotor, margin EBIT, ROE/ROA, tren 3–5 tahun terakhir
    - Pertumbuhan: Tinjau CAGR pendapatan, laba bersih, arus kas operasi
    - Kesehatan Neraca: Analisis struktur modal (DER, debt maturity), likuiditas (current ratio, quick ratio)
    - Dividen & Buyback: Riwayat yield, payout ratio, kebijakan manajemen

    3. Analisis Teknikal
    - Trend & Momentum: Interpretasi moving averages…
    - Volatilitas & Volume: ATR, Bollinger Bands, perubahan on‐balance volume
    - Level Kunci: Support/resistance jangka pendek, menengah, jangka panjang
    - Sinyal:
    {signals_str}
    4. Sintesis dan Stress Test
    - Tarik benang merah antara hasil fundamental & teknikal
    - Uji skenario negatif (e.g. suku bunga naik, laba melambat) dan positif (pertumbuhan sektor)
    - Diskusikan faktor risiko eksternal (geopolitik, regulasi, persaingan) dan mitigasinya

    5. Rekomendasi & Timeframe
    - Berikan sinyal BUY / HOLD / SELL beserta level entry, stop-loss, dan target profit
    - Sertakan horizon investasi (jangka pendek: 1–3 bulan, menengah: 6–12 bulan, panjang: >12 bulan)
    - Jelaskan asumsi utama dan sensitivitas output terhadap perubahan variabel kunci

    Gunakan data numerik konkret, tabel ringkasan, dan referensi teori analisis keuangan di sepanjang pembahasan. Buatlah penjelasan yang terstruktur, jelas, dan dapat langsung diimplementasikan oleh seorang portfolio manager."""
    
    
    ai_analysis = get_ai_analysis(deep_prompt)
    with st.spinner("Mengambil saran dari OBI…"):
        st.markdown("**Hasil Analisis AI AFA:**")
        st.write(ai_analysis)

    
    st.markdown("---")
    st.subheader("🎯 Analisis Teknikal Saham")
    period = st.selectbox("Pilih periode data:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
    df = yf.download(ticker, period=period)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['ticker'] = ticker
    stock_df = df.reset_index()  
    stock_df['date'] = pd.to_datetime(stock_df['Date'])
    stock_df.sort_values('date', inplace=True)

    stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
    stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
    stock_df['EMA20'] = stock_df['Close'].ewm(span=20, adjust=False).mean()

    delta = stock_df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    stock_df['RSI'] = 100 - (100 / (1 + rs))
    
    stock_df['EMA_short'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
    stock_df['EMA_long'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
    
    if stock_df['EMA_short'].iloc[-1] > stock_df['EMA_long'].iloc[-1]:
        double_ema_signal = "Bullish (EMA Short > EMA Long)"
    else:
        double_ema_signal = "Bearish (EMA Short <= EMA Long)"
    
    stock_df['STD20'] = stock_df['Close'].rolling(window=20).std()
    stock_df['Upper Band'] = stock_df['MA20'] + 2 * stock_df['STD20']
    stock_df['Lower Band'] = stock_df['MA20'] - 2 * stock_df['STD20']
    
    stock_df['MACD'] = stock_df['EMA_short'] - stock_df['EMA_long']
    stock_df['Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
    
    stock_df['PP'] = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
    stock_df['R1'] = 2 * stock_df['PP'] - stock_df['Low']
    stock_df['S1'] = 2 * stock_df['PP'] - stock_df['High']
    
    low14  = stock_df['Low'].rolling(window=14).min()
    high14 = stock_df['High'].rolling(window=14).max()
    stock_df['%K'] = (stock_df['Close'] - low14) / (high14 - low14) * 100
    stock_df['%D'] = stock_df['%K'].rolling(window=3).mean()
    
    latest = stock_df.iloc[-1]
    prev = stock_df.iloc[-2]
    signals = []
    
    if latest['Close'] < latest['Lower Band']:
        signals.append("Bollinger Bands : BELI - Harga di bawah Lower Band.")
    elif latest['Close'] > latest['Upper Band']:
        signals.append("Bollinger Bands : JUAL - Harga di atas Upper Band.")
    else:
        signals.append("Bollinger Bands : Tidak ada sinyal jelas.")

    if prev['MACD'] < prev['Signal'] and latest['MACD'] > latest['Signal']:
        signals.append("MACD : BELI - golden cross.")
    elif prev['MACD'] > prev['Signal'] and latest['MACD'] < latest['Signal']:
        signals.append("MACD : MACD: JUAL - Dead cross.")
    else:
        signals.append("MACD : Tidak ada sinyal silang.")

    if prev['%K'] < prev['%D'] and latest['%K'] > latest['%D']:
        signals.append("Stochastic : BELI – %K crosses above %D.")
    elif prev['%K'] > prev['%D'] and latest['%K'] < latest['%D']:
        signals.append("Stochastic : JUAL – %K crosses below %D.")
    else:
        signals.append("Stochastic : Tidak ada sinyal silang.")
    
    threshold = 0.01
    if abs(latest['Close'] - latest['S1']) / latest['Close'] < threshold:
        signals.append("Support : Dekat level support (BELI).")
    elif abs(latest['Close'] - latest['R1']) / latest['Close'] < threshold:
        signals.append("Resistance: Dekat level resistance (JUAL).")
    else:
        signals.append("Support/Resistance : Tidak ada sinyal khusus.")

    if latest['RSI'] > 70:
        signals.append("RSI: JUAL - Overbought.")
    elif latest['RSI'] < 30:
        signals.append("RSI: BELI - Oversold.")
    else:
        signals.append("RSI : Netral")
    
    avg_vol = stock_df['Volume'].rolling(window=20).mean().iloc[-1]
    if latest['Volume'] > avg_vol * 1.5:
        signals.append("Volume : BREAKOUT - Volume sangat tinggi.")
    elif latest['Volume'] < avg_vol * 0.5:
        signals.append("Volume : Sinyal - Volume sangat rendah.")
    else:
        signals.append("Volume : Volume normal.")
    
    for sig in signals:
        st.markdown(sig)
    st.write(f"Double EMA : {double_ema_signal}")

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Candlestick(x=stock_df['date'],
                                    open=stock_df['Open'],
                                    high=stock_df['High'],
                                    low=stock_df['Low'],
                                    close=stock_df['Close'],
                                    name='Candlestick'))
    fig_ma.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['MA20'], mode='lines',
                                name='MA20', line=dict(color='blue')))
    fig_ma.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['MA50'], mode='lines',
                                name='MA50', line=dict(color='orange')))
    fig_ma.update_layout(title=f'Double Moving Average (MA20 & MA50) - {ticker}',
                         xaxis_title='Tanggal', yaxis_title='Harga',
                         xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_ma, use_container_width=True)
    
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Candlestick(x=stock_df['date'],
                                     open=stock_df['Open'],
                                     high=stock_df['High'],
                                     low=stock_df['Low'],
                                     close=stock_df['Close'],
                                     name='Candlestick'))
    fig_ema.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['EMA_short'], mode='lines',
                                 name='EMA Short (12)', line=dict(color='magenta')))
    fig_ema.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['EMA_long'], mode='lines',
                                 name='EMA Long (26)', line=dict(color='brown')))
    fig_ema.update_layout(title=f'Double EMA (EMA Short & EMA Long) - {ticker}',
                          xaxis_title='Tanggal', yaxis_title='Harga',
                          xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_ema, use_container_width=True)
    
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Candlestick(x=stock_df['date'],
                                    open=stock_df['Open'],
                                    high=stock_df['High'],
                                    low=stock_df['Low'],
                                    close=stock_df['Close'],
                                    name='Candlestick'))
    fig_bb.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['Upper Band'], mode='lines', name='Upper Band', line=dict(color='red')))
    fig_bb.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
    fig_bb.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['Lower Band'], mode='lines', name='Lower Band', line=dict(color='green')))
    fig_bb.update_layout(title=f'Bollinger Bands - {ticker}', xaxis_title='Tanggal', yaxis_title='Harga', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_bb, use_container_width=True)
    
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['Signal'], mode='lines', name='Signal', line=dict(color='orange')))
    fig_macd.update_layout(title=f'MACD - {ticker}', xaxis_title='Tanggal', yaxis_title='Nilai')
    st.plotly_chart(fig_macd, use_container_width=True)

    fig_sr = go.Figure()
    fig_sr.add_trace(go.Candlestick(x=stock_df['date'],
                                    open=stock_df['Open'],
                                    high=stock_df['High'],
                                    low=stock_df['Low'],
                                    close=stock_df['Close'],
                                    name='Candlestick'))
    fig_sr.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['PP'], mode='lines', name='Pivot Point (PP)', line=dict(color='purple', dash='dash')))
    fig_sr.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['R1'], mode='lines', name='Resistance (R1)', line=dict(color='red', dash='dot')))
    fig_sr.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['S1'], mode='lines', name='Support (S1)', line=dict(color='green', dash='dot')))
    fig_sr.update_layout(title=f'Support & Resistance - {ticker}', xaxis_title='Tanggal', yaxis_title='Harga', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_sr, use_container_width=True)
    
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line=dict(dash='dash', color='red'), annotation_text='Overbought')
    fig_rsi.add_hline(y=30, line=dict(dash='dash', color='green'), annotation_text='Oversold')
    fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Tanggal', yaxis_title='RSI')
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['%K'],
                                   mode='lines', name='%K'))
    fig_stoch.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['%D'],
                                   mode='lines', name='%D'))
    fig_stoch.update_layout(
        title=f'Stochastic Oscillator (14,3) - {ticker}',
        xaxis_title='Tanggal',
        yaxis_title='Nilai',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig_stoch, use_container_width=True)
    
    fig_vol = px.bar(stock_df, x='date', y='Volume', title='Volume Perdagangan',
                     labels={'Volume': 'Volume', 'date': 'Tanggal'})
    st.plotly_chart(fig_vol, use_container_width=True)
    
with tab3:
    st.header("💬 AI Chat: Diskusi Finansial & Investasi")
    st.markdown("""
    Selamat datang di sesi chat AI tentang finansial dan investasi. 
    Ajukan pertanyaan seputar pasar saham, instrumen investasi, analisis fundamental, teknikal, dan lainnya.
    """)

    bubble_css = """
    <style>
      .chat-container {
          display: flex;
          flex-direction: column;
          margin-top: 20px;
      }
      .chat-bubble {
          max-width: 70%;
          padding: 10px 15px;
          border-radius: 20px;
          margin: 5px;
          font-size: 16px;
          line-height: 1.4;
      }
      .user-bubble {
          background-color: #1E90FF;
          align-self: flex-end;
          border-bottom-right-radius: 0;
      }
      .ai-bubble {
          background-color: #081F5C;
          align-self: flex-start;
          border-bottom-left-radius: 0;
      }
    </style>
    """
    
    st.markdown(bubble_css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state["chat_history"]:
        role = chat["role"]
        message = chat["message"]
        if role == "user":
            bubble = f'<div class="chat-bubble user-bubble"><strong>Anda:</strong> {message} </div>'
        else:
            bubble = f'<div class="chat-bubble ai-bubble"><strong>AI:</strong> {message} </div>'
        st.markdown(bubble, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.text_input("Tanyakan sesuatu tentang finansial atau investasi", key="user_input")

    if st.button("Kirim") and user_input:
        st.session_state["chat_history"].append({"role": "user", "message": user_input})

        def get_ai_response(prompt):
            try:
                model = genai.GenerativeModel(
                    model_name='models/gemini-2.0-flash',
                    system_instruction="Anda adalah asisten ahli finansial dan investasi yang akan memberikan analisis mendalam dan saran profesional."
                )
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2048
                    ),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                    }
                )
                
                return response.text
            except Exception as e:
                return f"Terjadi kesalahan: {str(e)}"
        
        ai_reply = get_ai_response(user_input)
        st.session_state["chat_history"].append({"role": "assistant", "message": ai_reply})
        st.rerun()
    
with tab4:
    st.header("📒 Portofolio Saya")
    port = st.session_state['portfolio']
    st.metric("Saldo Kas (IDR)", f"Rp{port['cash']:,.2f}")
    if port['positions']:
        rows=[]
        for tk,qty in port['positions'].items():
            try:
                pr,_,inf=get_stock_data(tk)
                fx2 = get_fx_rate_to_idr(inf.get('currency','USD')) or 1
                mv=qty*pr*fx2
            except: mv=0
            rows.append({'Ticker':tk,'Qty':qty,'Market Value (IDR)':mv})
        df=pd.DataFrame(rows)
        df['Market Value (IDR)']=df['Market Value (IDR)'].round(2)
        st.dataframe(df)
        pie_vals=df['Market Value (IDR)'].tolist()+[port['cash']]
        pie_lbl=df['Ticker'].tolist()+['Cash']
        fig=px.pie(names=pie_lbl,values=pie_vals,title='Alokasi Portofolio')
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("Belum ada posisi.")

st.markdown("---")
st.markdown("© 2025 - AFA | Math Wizard II")
