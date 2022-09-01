import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu


#Title y config pagina
st.set_page_config(
    page_title="Consumo de Energía Global y CO2 - Home",
    layout="wide"
)

st.title("Dashboard")
st.components.v1.iframe('https://app.powerbi.com/view?r=eyJrIjoiMDA3YTk0OTktYTE1Ni00Yzc1LTkyYTQtMmYzYWZmMjBhZDdlIiwidCI6IjQ1OGEwZGVkLWVlN2ItNDdjYy04MTg1LTJmM2Q2MDY1YjQ1MCJ9&pageName=ReportSection',
                        height=700)