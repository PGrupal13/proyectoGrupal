#Bibliotecas
import streamlit as st


#Title y config pagina
st.set_page_config(
    page_title="Global Energy consumption and CO2 - Home",
    layout="wide"
)


st.title('Global Energy consumption and CO2')

st.markdown("""
            <div style="text-align: justify;">
            Climate change has accelerated to unprecedented levels as a result of human activities, one of the main culprits being the need for energy obtained from various fossil fuel sources. The impact of energy development on the environment and the consumption generated, attract companies to take action on how to intervene in these problems. Which leads to measurements of consumption and generation to intervene or improve said generation/consumption.

            The United States Environmental Protection Agency (EPA) estimates that today almost 33% of CO₂ emissions in the United States are generated due to energy production, due to the use of fossil fuels to produce energy. Likewise, the International Energy Agency (IEA) estimates that by 2014, 49% of the carbon dioxide emissions emitted into the atmosphere were produced from the burning of fossil fuels for heating and power generation.

            Today it is estimated that the concentration of CO₂ in the atmosphere is 414.37 ppm, which makes it more necessary to apply emission reduction strategies. In the case of energy production, it has been proposed to make a transition from fossil energy sources to clean and renewable energy sources. However, the application and use of these energy sources also lead to CO₂ emissions before, during and after their application.

            Additionally, treatments have been carried out in order to reduce the concentration of CO₂ in the atmosphere. The Kyoto Protocol, signed on December 11, 1997 and in force since February 16, 2005, aims to reduce emissions of six greenhouse gases by 5% for the period 2008-2012 compared to 1990, expanding the period until 2020. For its part, the Paris Agreement, signed on April 22, 2016 and in force since November 4, 2016, aims to keep the increase in average global temperature below 2 °C compared to pre-industrial values ​​and make efforts to limit the increase to 1.5 °C by reducing greenhouse gas emissions.

            With this analysis we want to evaluate the amount of carbon dioxide emissions throughout the time series, assessing the impacts of the Kyoto and Paris agreements after their implementation. As well as the economic opportunity thanks to the implementation of carbon credits


            </div>
            """, unsafe_allow_html=True)