import streamlit as st


#Title y config pagina
st.set_page_config(
    page_title="Consumo de Energía Global y CO2 - Home",
    layout="wide"
)

st.title("Dashboard")
@st.experimental_memo
def load_dashboard():
    return 'https://app.powerbi.com/view?r=eyJrIjoiMDA3YTk0OTktYTE1Ni00Yzc1LTkyYTQtMmYzYWZmMjBhZDdlIiwidCI6IjQ1OGEwZGVkLWVlN2ItNDdjYy04MTg1LTJmM2Q2MDY1YjQ1MCJ9&pageName=ReportSection'

st.components.v1.iframe(load_dashboard(), height=700)