import streamlit as st


#Title y config pagina
st.set_page_config(
    page_title="Global Energy consumption and CO2 - Home",
    layout="wide"
)

st.title("Dashboard")
@st.experimental_memo
def load_dashboard():
    return 'https://app.powerbi.com/view?r=eyJrIjoiOTRjNWFmNjctOWE2NS00NWEwLWI0MjEtZjJiYjBiYzM0NGI2IiwidCI6IjQ1OGEwZGVkLWVlN2ItNDdjYy04MTg1LTJmM2Q2MDY1YjQ1MCJ9'

st.components.v1.iframe(load_dashboard(), height=700)