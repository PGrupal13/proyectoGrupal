import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    choose = option_menu("Menú", ['Francia Vs Alemania Energía Nuclear',
                                  'Insight 2',
                                  'Insight 3'])