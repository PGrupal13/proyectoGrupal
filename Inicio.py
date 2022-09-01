#Bibliotecas
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


st.title('Consumo de Energía Global y CO2')

st.markdown("""
            <div style="text-align: justify;">
            El cambio climático se ha acelerado a niveles sin precedentes como consecuencia de las actividades humanas, siendo una 
            de las mayores responsables la necesidad de energía obtenida a partir de diversas fuentes de combustibles fósiles. El 
            impacto del desarrollo energético en el ambiente y los consumos generados, atraen a las compañías a tomar acción en cómo 
            intervenir en estas problemáticas. Lo cual lleva a mediciones de consumo y generación para intervenir o mejorar dicha 
            generación/consumo.

            La Agencia de Protección ambiental de los Estados Unidos (EPA) estima que hoy en día casi un 33% de las emisiones de CO₂ 
            en los Estados Unidos son generadas debido a la producción energética, esto debido al uso de combustibles fósiles para 
            producir energía. Asimismo, la Agencia Internacional de Energía (IEA) estima que para el 2014 el 49% de las emisiones de 
            dióxido de carbono emitidas a la atmósfera se produjeron a partir de la quema de combustibles fósiles para la generación 
            de calefacción y energía.

            Al día de hoy se estima que la concentración de CO₂ en la atmósfera es de 414.37 ppm, esto hace que sea más necesario 
            aplicar estrategias de reducción de emisiones. En el caso de la producción energética se ha propuesto realizar una 
            transición de fuentes de energía fósil hacia fuentes de energía limpia y renovables. Sin embargo, la aplicación y uso de 
            dichas fuentes de energía también conllevan emisiones de CO₂ antes, durante y después de su aplicación.

            Adicionalmente, se han realizado tratados con el fin de disminuir la concentración de CO₂ en la atmósfera. El Protocolo 
            de Kioto, firmado el 11 de diciembre de 1997 y vigente desde el 16 de febrero de 2005 tiene como objetivo reducir las 
            emisiones de seis gases de efecto invernadero en un 5% para el periodo 2008-2012 en comparación con 1990, ampliando el 
            periodo hasta el 2020. Por su lado el acuerdo de París, firmado el 22 de abril de 2016 y en vigencia desde el 4 de 
            noviembre de 2016 tiene como objetivo mantener el aumento de la temperatura global promedio por debajo de los 2 °C 
            comparando con los valores preindustriales y realizar esfuerzos para limitar el aumento a 1,5 °C mediante la reducción 
            de emisiones de gases de efecto invernadero.

            Con este análisis se quiere Evaluar la cantidad de emisiones de dióxido de carbono a lo largo de la serie de tiempo, 
            valorando los impactos de los acuerdos de Kioto y París después de su implementación. Así como la oportunidad económica 
            gracias a la implementación de bonos de carbono.

            </div>
            """, unsafe_allow_html=True)