# SmartSensor Industrial Dashboard (Streamlit)

Simulación de sensores (temperatura/humedad) con filtros digitales (Media, Mediana, Exponencial, Savitzky–Golay) y alertas por email vía SMTP.

## Correr local
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Opcional: credenciales locales
mkdir -p .streamlit
# crear .streamlit/secrets.toml con SMTP_USER, SMTP_PASS, ALERT_TO, etc.

streamlit run streamlit_app.py
