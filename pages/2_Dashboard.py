import streamlit as st


#Title y config pagina
st.set_page_config(
    page_title="Global Energy consumption and CO2 - Home",
    layout="wide"
)

st.title("Dashboard")
@st.experimental_memo
def load_dashboard():
    return 'https://app.powerbi.com/view?r=eyJrIjoiZDI2MTlhNDctN2M2OS00OTk3LWJiZDEtNjg4OWM3YjRiMDZmIiwidCI6IjQ1OGEwZGVkLWVlN2ItNDdjYy04MTg1LTJmM2Q2MDY1YjQ1MCJ9&pageName=ReportSection'

st.components.v1.iframe(load_dashboard(), height=700)