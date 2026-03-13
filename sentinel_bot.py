import yfinance as yf
import pandas as pd
import requests
import os
import datetime

# Credenciales de Telegram (se cargarán desde GitHub Secrets)
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main():
    # Descargar datos de los últimos 6 meses
    tickers = ["SPY", "^VIX"]
    data = yf.download(tickers, period="6mo", progress=False)
    
    # Extraer Cierres
    closes = data['Close'].ffill()
    spy_close = closes['SPY'].iloc[-1]
    vix_close = closes['^VIX'].iloc[-1]
    
    # Calcular Indicadores
    sma20 = closes['SPY'].rolling(window=20).mean().iloc[-1]
    sma50 = closes['SPY'].rolling(window=50).mean().iloc[-1]
    
    # Calcular RSI
    closes_spy_full = closes['SPY'].dropna()
    rsi_val = calculate_rsi(closes_spy_full).iloc[-1]
    
    # Lógica DCA Institucional
    zona_verde_oscuro = (spy_close < sma50 and rsi_val < 40) or (vix_close > 25) or (rsi_val < 30)
    zona_verde_claro = (not zona_verde_oscuro) and ((spy_close < sma20) or (rsi_val < 50))
    zona_neutra = (not zona_verde_oscuro) and (not zona_verde_claro)
    
    if zona_verde_oscuro:
        accion = "🔴 *PÁNICO: COMPRA AGRESIVA*\nInyectar cuota TRIPLE (Romper el chanchito)."
    elif zona_verde_claro:
        accion = "🟡 *DESCUENTO: COMPRA REFORZADA*\nInyectar cuota DOBLE."
    else:
        accion = "🟢 *TENDENCIA: APORTE REGULAR*\nInyectar 1 cuota normal."

    # Construir Mensaje
    fecha = datetime.datetime.now().strftime("%Y-%m-%d")
    mensaje = f"🤖 **Sentinel DCA Report** | {fecha}\n\n"
    mensaje += f"📊 **SPY:** ${spy_close:.2f}\n"
    mensaje += f"📈 **RSI (14):** {rsi_val:.1f}\n"
    mensaje += f"😨 **VIX:** {vix_close:.1f}\n"
    mensaje += f"📉 **SMA 50:** ${sma50:.2f}\n\n"
    mensaje += f"🎯 **DICTAMEN:**\n{accion}"
    
    # Enviar
    send_telegram_message(mensaje)
    print("Reporte enviado exitosamente.")

if __name__ == "__main__":
    main()
