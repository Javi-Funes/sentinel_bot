import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import json
import datetime
import logging

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Credenciales ──────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ─── Historial ─────────────────────────────────────────────────────────────────
HISTORIAL_PATH = "historial.json"
MAX_HISTORIAL  = 5

CUOTA_ORDEN = ["SIMPLE", "DOBLE", "TRIPLE", "DOBLE (Degradada de Triple)", "SIMPLE (Degradada de Doble)"]
CUOTA_NIVEL = {
    "SIMPLE":                    1,
    "SIMPLE (Degradada de Doble)": 1,
    "DOBLE":                     2,
    "DOBLE (Degradada de Triple)": 2,
    "TRIPLE":                    3,
}


def cargar_historial() -> list:
    if not os.path.exists(HISTORIAL_PATH):
        return []
    try:
        with open(HISTORIAL_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"No se pudo leer historial: {e}")
        return []


def guardar_historial(historial: list, fecha: str, cuota: str):
    historial.append({"fecha": fecha, "cuota": cuota})
    historial = historial[-MAX_HISTORIAL:]          # conservar solo los últimos N
    try:
        with open(HISTORIAL_PATH, "w") as f:
            json.dump(historial, f)
        log.info(f"Historial guardado ({len(historial)} entradas).")
    except Exception as e:
        log.warning(f"No se pudo guardar historial: {e}")


def formatear_historial(historial: list, cuota_hoy: str) -> str:
    if not historial:
        return ""

    FLECHA = {-1: " ⬇️", 0: "", 1: " ⬆️"}

    lineas = []
    entradas = historial[-(MAX_HISTORIAL - 1):]     # máximo 4 previas + hoy = 5

    prev_nivel = None
    for entrada in entradas:
        nivel = CUOTA_NIVEL.get(entrada["cuota"], 1)
        flecha = ""
        if prev_nivel is not None:
            diff = nivel - prev_nivel
            flecha = FLECHA.get(1 if diff > 0 else (-1 if diff < 0 else 0), "")
        lineas.append(f"  {entrada['fecha']} → {entrada['cuota']}{flecha}")
        prev_nivel = nivel

    # Agregar el día de hoy con la flecha comparada al último registro
    nivel_hoy = CUOTA_NIVEL.get(cuota_hoy, 1)
    flecha_hoy = ""
    if prev_nivel is not None:
        diff = nivel_hoy - prev_nivel
        flecha_hoy = FLECHA.get(1 if diff > 0 else (-1 if diff < 0 else 0), "")
    lineas.append(f"  {datetime.datetime.now().strftime('%Y-%m-%d')} → {cuota_hoy}{flecha_hoy} ← hoy")

    return "📅 *Historial reciente:*\n" + "\n".join(lineas)


# ─── Telegram ──────────────────────────────────────────────────────────────────
def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("Faltan credenciales de Telegram (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID).")
        return
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        log.info("Mensaje enviado a Telegram.")
    except Exception as e:
        log.error(f"Error al enviar a Telegram: {e}")


# ─── Indicadores ───────────────────────────────────────────────────────────────
def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Wilder (EWM). Mismo resultado que TradingView con 'RMA'."""
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).ewm(alpha=1 / window, adjust=False).mean()
    loss  = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / window, adjust=False).mean()
    rs    = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index. Preserva el índice del DataFrame original."""
    tp    = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf   = tp * df["Volume"]
    diff  = tp.diff()

    pos_mf = rmf.where(diff > 0, 0.0)
    neg_mf = rmf.where(diff < 0, 0.0)

    pos_sum = pos_mf.rolling(window=window).sum()
    neg_sum = neg_mf.rolling(window=window).sum().replace(0, 1e-10)

    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


# ─── Descarga de datos ─────────────────────────────────────────────────────────
def descargar_ticker(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Descarga un ticker y normaliza columnas ante MultiIndex de yfinance >= 0.2."""
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    fecha = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. Descargar datos
    log.info("Descargando datos...")
    try:
        df_spy = descargar_ticker("SPY",  period="1y")
        df_iwm = descargar_ticker("IWM",  period="1y")
        df_vix = descargar_ticker("^VIX", period="3mo")
    except Exception as e:
        msg = f"⚠️ *Sentinel DCA* | {fecha}\nFallo en descarga de datos: `{e}`"
        send_telegram_message(msg)
        log.error(f"Descarga fallida: {e}")
        return

    # 2. Validar que no estén vacíos
    for nombre, df in [("SPY", df_spy), ("IWM", df_iwm), ("VIX", df_vix)]:
        if df.empty:
            msg = f"⚠️ *Sentinel DCA* | {fecha}\nSin datos para {nombre}. Mercado cerrado o error en API."
            send_telegram_message(msg)
            log.error(f"DataFrame vacío: {nombre}")
            return

    # 3. Extraer valores clave
    vix_close = df_vix["Close"].ffill().iloc[-1]
    if pd.isna(vix_close):
        send_telegram_message(f"⚠️ *Sentinel DCA* | {fecha}\nVIX devolvió NaN. Señal no confiable, abortando.")
        return

    spy_close = df_spy["Close"].ffill().iloc[-1]
    spy_vol   = df_spy["Volume"].ffill().iloc[-1]
    avg_vol   = df_spy["Volume"].rolling(20).mean().iloc[-1]

    # 4. Indicadores técnicos SPY
    sma20   = df_spy["Close"].rolling(20).mean().iloc[-1]
    sma50   = df_spy["Close"].rolling(50).mean().iloc[-1]
    rsi_val = calculate_rsi(df_spy["Close"]).iloc[-1]
    mfi_spy = calculate_mfi(df_spy).iloc[-1]

    # 5. Indicadores IWM (canario)
    mfi_iwm    = calculate_mfi(df_iwm).iloc[-1]
    climax_vol = spy_vol > (avg_vol * 1.5)

    log.info(f"SPY={spy_close:.2f} | VIX={vix_close:.1f} | RSI={rsi_val:.1f} | MFI_SPY={mfi_spy:.1f} | MFI_IWM={mfi_iwm:.1f}")

    # 6. Lógica DCA — zonas explícitas sin solapamiento
    #    Pánico: SPY bajo SMA50 + RSI bajo 40, o VIX alto, o RSI extremo
    zona_verde_oscuro = (
        (spy_close < sma50 and rsi_val < 40)
        or (vix_close > 25)
        or (rsi_val < 30)
    )
    #    Descuento: fuera de pánico, pero SPY bajo SMA20 o RSI entre 40-50
    zona_verde_claro = (
        not zona_verde_oscuro
        and (spy_close < sma20 or (40 <= rsi_val < 50))
    )

    # 7. Acción operativa con filtro IWM
    alerta_iwm    = ""
    detalle_accion = ""

    if zona_verde_oscuro:
        if mfi_iwm < 30:
            status  = "🔴 PÁNICO CON RIESGO SISTÉMICO"
            cuota   = "DOBLE (Degradada de Triple)"
            alerta_iwm     = "⚠️ *Alerta IWM:* Small Caps colapsando (MFI < 30). Riesgo sistémico activo."
            detalle_accion = (
                "Ejecuta 2 cuotas en CEDEAR SPY. Aplica TWAP fraccionando la orden "
                "en los próximos 3 días (12:30 a 15:00 hs). Conserva la 3ra cuota en liquidez."
            )
        else:
            status  = "🔴 PÁNICO (Oportunidad Fuerte)"
            cuota   = "TRIPLE"
            detalle_accion = (
                "Inyecta 3 cuotas en CEDEAR SPY. Mercado amplio resiste bien el pánico. "
                "Orden completa o fraccionada en 2 días con órdenes límite."
            )
    elif zona_verde_claro:
        if mfi_iwm < 35:
            status  = "🟡 DESCUENTO CON DEBILIDAD"
            cuota   = "SIMPLE (Degradada de Doble)"
            alerta_iwm     = "⚠️ *Alerta IWM:* Divergencia negativa. SPY retrocede pero Small Caps sufren más."
            detalle_accion = (
                "Inyecta solo 1 cuota regular en CEDEAR SPY. No gastes liquidez extra; "
                "la amplitud del mercado es débil y el descuento podría profundizarse."
            )
        else:
            status  = "🟡 DESCUENTO"
            cuota   = "DOBLE"
            detalle_accion = (
                "Inyecta 2 cuotas en CEDEAR SPY. Corrección sana sin impacto en mercado amplio. "
                "Divide la orden en dos tandas para promediar el spread en BYMA."
            )
    else:
        status  = "🟢 TENDENCIA"
        cuota   = "SIMPLE"
        detalle_accion = (
            "Inyecta 1 cuota regular en CEDEAR SPY. Flujo normal. "
            "Ejecuta orden límite cercana al precio Ask."
        )

    # Nota de volumen climático (informativa)
    nota_vol = "\n📢 *Nota:* Volumen climático detectado (>1.5x promedio). Señal de capitulación posible." if climax_vol else ""

    # 8. Historial
    historial      = cargar_historial()
    bloque_historial = formatear_historial(historial, cuota)

    # 9. Construir mensaje
    # Nota: se escapan los $ para evitar conflictos con Markdown v1 de Telegram
    mensaje  = f"🤖 *Sentinel DCA Report* | {fecha}\n"
    mensaje += f"🎯 *Activo:* CEDEAR SPY (BYMA)\n\n"
    mensaje += f"📊 *Precio SPY:* USD {spy_close:.2f}\n"
    mensaje += f"😨 *VIX:* {vix_close:.1f}\n"
    mensaje += f"📈 *RSI SPY:* {rsi_val:.1f}\n"
    mensaje += f"🌊 *MFI SPY:* {mfi_spy:.1f}\n"
    mensaje += f"🦅 *MFI IWM (Small Caps):* {mfi_iwm:.1f}\n"
    mensaje += nota_vol + "\n\n"

    if alerta_iwm:
        mensaje += f"{alerta_iwm}\n\n"

    mensaje += f"⚖️ *ESTADO:* {status}\n"
    mensaje += f"💰 *CUOTA SUGERIDA:* *{cuota}*\n\n"
    mensaje += f"📋 *DETALLE OPERATIVO:*\n{detalle_accion}\n"

    if bloque_historial:
        mensaje += f"\n{bloque_historial}"

    # 10. Enviar y guardar historial
    send_telegram_message(mensaje)
    guardar_historial(historial, fecha, cuota)
    log.info(f"Ejecución completada. Estado: {status} | Cuota: {cuota}")


if __name__ == "__main__":
    main()
