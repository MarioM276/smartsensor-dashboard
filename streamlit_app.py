# ================================================================
# SmartSensor Industrial Dashboard (Streamlit)
# Simulación + Filtros + Alertas por Email (SMTP)
# ================================================================

import os
import ssl
import smtplib
import time
import random
from email.message import EmailMessage
from collections import deque

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import streamlit as st
import plotly.graph_objs as go

# ------------------- Config de página -------------------
st.set_page_config(page_title="SmartSensor Dashboard", page_icon="📈", layout="wide")

APP_TITLE = "SmartSensor Industrial Dashboard (Streamlit)"
SAMPLE_PERIOD_MS = 1000          # refresco de 1 s
DEFAULT_BUFFER = 180
ALERT_COOLDOWN_S = 60            # anti-spam de alertas (s)

# ------------------- Helpers de credenciales -------------------
def get_secret(key: str, default: str = "") -> str:
    # 1) st.secrets (Streamlit Cloud)  2) variables de entorno (local)
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

SMTP_HOST = get_secret("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(get_secret("SMTP_PORT", "465"))
SMTP_USER = get_secret("SMTP_USER", "")
SMTP_PASS = get_secret("SMTP_PASS", "")
ALERT_TO  = [m.strip() for m in get_secret("ALERT_TO", "").split(",") if m.strip()]

# ------------------- Estado (buffers) -------------------
if "t" not in st.session_state:
    st.session_state.t = deque(maxlen=DEFAULT_BUFFER)
    st.session_state.temp = deque(maxlen=DEFAULT_BUFFER)
    st.session_state.hum  = deque(maxlen=DEFAULT_BUFFER)
    st.session_state.last_alert_ts = 0.0

# ------------------- Funciones core -------------------
def notify_email(subject: str, body: str) -> str:
    missing = []
    if not SMTP_USER: 
        missing.append("SMTP_USER")
    if not SMTP_PASS: 
        missing.append("SMTP_PASS")
    if not ALERT_TO:  
        missing.append("ALERT_TO")
    if missing:
        return f"Email ✗ (faltan vars: {', '.join(missing)})"
    try:
        msg = EmailMessage()
        msg["From"] = SMTP_USER
        msg["To"]   = ", ".join(ALERT_TO)
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return "Email ✓"
    except Exception as e:
        return f"Email ✗ {type(e).__name__}: {e}"

def filtro_media(x, w):
    w = max(1, int(w))
    y = np.convolve(x, np.ones(w)/w, mode="same")
    return y[:len(x)]

def filtro_mediana(x, w):
    w = max(1, int(w) | 1)  # impar
    y = pd.Series(x).rolling(window=w, center=True, min_periods=1).median().to_numpy()
    return y[:len(x)]

def filtro_exp(x, alpha):
    alpha = float(np.clip(alpha, 0.01, 0.99))
    y = np.zeros_like(x, dtype=float)
    if len(x) == 0: 
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def filtro_savgol(x, w, p):
    if len(x) < 3:
        return np.array(x, dtype=float)
    w = max(3, int(w) | 1)  # impar
    if w >= len(x):
        w = len(x)-1 if (len(x)-1) % 2 == 1 else len(x)-2
    p = int(np.clip(p, 1, 5))
    if w <= p:
        w = p+2 if (p+2) % 2 == 1 else p+3
    try:
        y = savgol_filter(np.asarray(x, dtype=float), window_length=w, polyorder=p, mode="interp")
        return y[:len(x)]
    except Exception:
        return np.asarray(x, dtype=float)

def sim_lectura_temp(n):
    base = 25 + 0.03*n
    ruido = random.uniform(-1.0, 1.0)
    pico  = 8.0 if random.random() < 0.02 else 0.0
    return base + ruido + pico

def sim_lectura_hum(n):
    base = 55 + 0.02*n
    ruido = random.uniform(-2.0, 2.0)
    pico  = 25.0 if random.random() < 0.015 else 0.0
    return max(0, min(100, base + ruido + pico))

def chequear_alertas(temp, hum, th):
    now = time.time()
    if not th["alerts_on"]:
        return ""
    if now - st.session_state.last_alert_ts < ALERT_COOLDOWN_S:
        return ""
    motivos = []
    if temp >= th["temp_crit"]:
        motivos.append(f"Temp CRÍTICA ({temp:.1f}°C ≥ {th['temp_crit']}°C)")
    elif temp >= th["temp_warn"]:
        motivos.append(f"Temp ALTA ({temp:.1f}°C ≥ {th['temp_warn']}°C)")
    if hum >= th["hum_crit"]:
        motivos.append(f"Humedad CRÍTICA ({hum:.1f}% ≥ {th['hum_crit']}%)")
    elif hum >= th["hum_warn"]:
        motivos.append(f"Humedad ALTA ({hum:.1f}% ≥ {th['hum_warn']}%)")
    if motivos:
        body = (
            "Se detectaron condiciones de riesgo:\n"
            + "\n".join(f"- {m}" for m in motivos)
            + f"\n\nTimestamp: {pd.Timestamp.now()}\nSistema: {APP_TITLE}"
        )
        status = notify_email("⚠️ SmartSensor: Alerta de Condición Crítica", body)
        st.session_state.last_alert_ts = now
        return status
    return ""

# ------------------- Sidebar (controles) -------------------
st.sidebar.title("Parámetros")

buf   = st.sidebar.slider("Buffer", 60, 600, DEFAULT_BUFFER, 20)
maw   = st.sidebar.slider("Media móvil (w)", 3, 21, 5, 1)
medw  = st.sidebar.slider("Mediana (w)", 3, 21, 5, 1)
alpha = st.sidebar.slider("Exponencial (α)", 0.05, 0.95, 0.30, 0.05)
sgw   = st.sidebar.slider("Savitzky (ventana)", 5, 31, 7, 2)
sgp   = st.sidebar.slider("Savitzky (orden)", 1, 5, 2, 1)

st.sidebar.markdown("---")
alerts_on = st.sidebar.toggle("Alertas activas", value=True)
col1, col2 = st.sidebar.columns(2)
temp_warn = col1.number_input("Temp warn (°C)", value=35.0, step=0.5)
temp_crit = col2.number_input("Temp crit (°C)", value=40.0, step=0.5)
col3, col4 = st.sidebar.columns(2)
hum_warn  = col3.number_input("Hum warn (%)", value=70.0, step=1.0)
hum_crit  = col4.number_input("Hum crit (%)", value=85.0, step=1.0)

st.sidebar.markdown("---")
filters_on = st.sidebar.multiselect(
    "Mostrar filtros", ["Media", "Mediana", "Exp", "Savitzky"],
    default=["Media", "Mediana", "Exp", "Savitzky"]
)
auto = st.sidebar.toggle("Auto-refresh (1 s)", value=True)

if st.sidebar.button("Probar Alerta (Email)"):
    st.sidebar.success(notify_email("🧪 Test SmartSensor", f"Prueba {pd.Timestamp.now()}"))

# Ajustar maxlen si cambió el buffer
if st.session_state.t.maxlen != buf:
    st.session_state.t = deque(st.session_state.t, maxlen=buf)
    st.session_state.temp = deque(st.session_state.temp, maxlen=buf)
    st.session_state.hum  = deque(st.session_state.hum,  maxlen=buf)

# Auto-refresh SIN streamlit_extras (simple y compatible)
if auto:
    time.sleep(SAMPLE_PERIOD_MS / 1000.0)
    st.rerun()

# ------------------- Simulación (un tick por render) -------------------
n = (st.session_state.t[-1] + 1) if len(st.session_state.t) else 0
st.session_state.t.append(n)
st.session_state.temp.append(sim_lectura_temp(n))
st.session_state.hum.append(sim_lectura_hum(n))

x    = np.arange(len(st.session_state.t))
temp = np.array(st.session_state.temp, dtype=float)
hum  = np.array(st.session_state.hum,  dtype=float)

# ------------------- Filtros -------------------
temp_ma  = filtro_media(temp, maw)
temp_med = filtro_mediana(temp, medw)
temp_exp = filtro_exp(temp, alpha)
temp_sg  = filtro_savgol(temp, sgw, sgp)

hum_ma  = filtro_media(hum, maw)
hum_med = filtro_mediana(hum, medw)
hum_exp = filtro_exp(hum, alpha)
hum_sg  = filtro_savgol(hum, sgw, sgp)

# ------------------- Layout principal -------------------
st.title(APP_TITLE)
st.caption("Simulación de temperatura y humedad con filtros digitales + alerta por email.")

kpi1, kpi2, status_box = st.columns([1, 1, 2])
kpi1.metric("Temperatura (°C)", f"{temp[-1]:.1f}")
kpi2.metric("Humedad (%)", f"{hum[-1]:.1f}")

thresholds = dict(
    alerts_on=alerts_on,
    temp_warn=temp_warn, temp_crit=temp_crit,
    hum_warn=hum_warn,   hum_crit=hum_crit,
)
status = chequear_alertas(temp[-1], hum[-1], thresholds)
if status:
    status_box.success(status)

colA, colB = st.columns(2)

# ----- Gráfico Temperatura -----
fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=x, y=temp, name="Sensor", mode="lines+markers"))
if "Media" in filters_on:   
    fig_t.add_trace(go.Scatter(x=x, y=temp_ma,  name="Media"))
if "Mediana" in filters_on: 
    fig_t.add_trace(go.Scatter(x=x, y=temp_med, name="Mediana"))
if "Exp" in filters_on:     
    fig_t.add_trace(go.Scatter(x=x, y=temp_exp, name="Exp"))
if "Savitzky" in filters_on:
    fig_t.add_trace(go.Scatter(x=x, y=temp_sg,  name="Savitzky"))
fig_t.add_hline(y=temp_warn, line_dash="dot", line_color="orange", annotation_text="Warn")
fig_t.add_hline(y=temp_crit, line_dash="dot", line_color="red",    annotation_text="Crit")
fig_t.update_layout(template="plotly_dark", height=360, margin=dict(l=30, r=20, t=30, b=30),
                    legend=dict(orientation="h"))
colA.plotly_chart(fig_t, use_container_width=True)

# ----- Gráfico Humedad -----
fig_h = go.Figure()
fig_h.add_trace(go.Scatter(x=x, y=hum, name="Sensor", mode="lines+markers"))
if "Media" in filters_on:   
    fig_h.add_trace(go.Scatter(x=x, y=hum_ma,  name="Media"))
if "Mediana" in filters_on: 
    fig_h.add_trace(go.Scatter(x=x, y=hum_med, name="Mediana"))
if "Exp" in filters_on:     
    fig_h.add_trace(go.Scatter(x=x, y=hum_exp, name="Exp"))
if "Savitzky" in filters_on:
    fig_h.add_trace(go.Scatter(x=x, y=hum_sg,  name="Savitzky"))
fig_h.add_hline(y=hum_warn, line_dash="dot", line_color="orange", annotation_text="Warn")
fig_h.add_hline(y=hum_crit, line_dash="dot", line_color="red",    annotation_text="Crit")
fig_h.update_layout(template="plotly_dark", height=360, margin=dict(l=30, r=20, t=30, b=30),
                    legend=dict(orientation="h"))
colB.plotly_chart(fig_h, use_container_width=True)

st.info("En la nube, carga credenciales en *Settings → Secrets*. Localmente usa `.streamlit/secrets.toml`.")
