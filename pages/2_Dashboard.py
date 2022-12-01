import streamlit as st


#Title y config pagina
st.set_page_config(
    page_title="Global Energy consumption and CO2 - Home",
    layout="wide"
)

st.title("Dashboard")
@st.experimental_memo
def load_dashboard():
    return 'https://app.powerbi.com/reportEmbed?reportId=16803e42-658f-48ac-bd2b-e2f1c8bd36a9&autoAuth=true&ctid=458a0ded-ee7b-47cc-8185-2f3d6065b450'

st.components.v1.iframe(load_dashboard(), height=700)