import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
    choose = option_menu("Menú", ['Francia Vs Alemania Energía Nuclear',
                                  'Insight 2',
                                  'Insight 3'])



from apiathena import apirequest 
import pandas as pd
import matplotlib.pyplot as plt

query01='''
select * from  main_db.typesplit2
where year > 1990 ;
'''
query02='''
SELECT year, country_code,energy_type_code, energy_consumption, co2_emission
FROM "main_db"."energyco2" 
where country_code in ('DEU', 'FRA')and year > 1991 ;
'''

dfq = apirequest(query01)  ############# llamado api
dfq2 = apirequest(query02)

def relative_energy_consumption(cou_cod, country_name):
    
    df= dfq.loc[dfq.country_code == cou_cod]
    df_group= df.groupby(['year']).sum()
    
    df_group = df_group.divide(df_group.sum(axis=1), axis=0)
    
    y1=df_group['1']
    y2=df_group['2']
    y3=df_group['3']
    y4=df_group['4']
    y5=df_group['5']
    #y6 = df_totalemision_group['totalemision']
    
    fig = plt.figure(figsize = (12, 6), dpi=120)
    plt.title('Relative energy consumption by type- ' + country_name, fontsize=16)
    plt.stackplot(df_group.index,  y1 , y2, y4, y3, y5, labels=['coal','natural gas', 'petroleum and other liquids',
                'nuclear', 'renewables and others'])
        
    plt.xlabel("Year")
    plt.ylabel("Energy consumption ")
    plt.axvline(x=2005, color='black', label='Kioto protocol')
    plt.axvline(x=2016, color='gray', label = 'Paris agreement')
    
    plt.grid(visible=True, axis='y')
    plt.legend(loc='upper left', prop={'size': 12})
    st.pyplot(fig)
    plt.show()
    
def compare():    
    df_group= dfq2.groupby(['country_code', 'year']).sum().reset_index()

    df_group.dropna(inplace=True)
    df_group['indice'] = df_group.apply(lambda x: x['co2_emission'] / x['energy_consumption'], axis=1 )

    df_fra = df_group.loc[df_group['country_code'] == 'FRA']
    df_deu =  df_group.loc[df_group['country_code'] == 'DEU']
    x=df_fra.year


    fig = plt.figure(figsize = (12, 6), dpi=120)
    plt.title('CO2 emission per unit of energy consumed, total', fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("million ton / Quad BTU")
    plt.plot (x, df_fra.indice, label = "FRANCE ")
    plt.plot (x, df_deu.indice, label = "GERMANY ")

    plt.grid()
    
    plt.legend()
    st.pyplot(fig)
    plt.show()
    
    
    
st.header('Insights')
st.subheader('Comparativa de emisiones por tipo de energia consumida Alemania-Francia ')

relative_energy_consumption('DEU', 'Germany')
relative_energy_consumption('FRA', 'France')

st.markdown('''Se observa como Alemania pese a [incrementar la producción de energía eólica-solar](https://www.worldenergytrade.com/politica/europa/alemania-aumenta-su-objetivo-de-energia-limpia-del-65-al-80-para-2030) no ha bajado significativamente
            en consumo de fuentes fósiles(80% en 2019) sobre todo de carbón. Por otra parte Francia, que mantiene una cuota alta en 
            energía nuclear, logra reducir proporcionalmente su consumo fósil (<50%)''')
st.markdown('Lo anterior se refleja en una gran diferencia de emisiones de CO2 por unidad de energia consumida:')
compare()                      
         

st.markdown(''' La explicacion posiblemente sea la naturaleza intermitente de las energias solar y eolica(no producen de noche, dependen del clima, geografia  y estaciones)
            lo que obliga a suplir la falencia con el respaldo de fuentes convencionales, lo que tambien aumenta los costos. 
            ''')


st.subheader('Conclusion')
st.markdown('''De lo expuesto se sugiere incrementar las fuentes de energia nuclear para reducir las emisiones de 
               efecto invernadero.''')
             





