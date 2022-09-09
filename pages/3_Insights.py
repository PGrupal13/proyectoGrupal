import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from get_df_co2aprox import *
from apiathena import apirequest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from streamlit_option_menu import option_menu
import streamlit.components as stc
import base64 
import time
import joblib
from asyncore import write
import pydeck as pdk
import folium
from streamlit_folium import st_folium


@st.experimental_memo(ttl=86400)
def request(query):
    return apirequest(query)

with st.sidebar:
    choose = option_menu("Insights & Predictions", ['France Vs Germany Nuclear Energy',
                                  'Countries Clusters by relationship between GDP and CO2 Emission',
                                  'Predict the CO2 emission using the share percentage of energy production',
                                  'Forecasting consumption and emission',
                                  'CO2 prediction with Neural Network',
                                  'CO2 emissions (approximation)',
                                  'Predict the emission of a house in a country'])


if choose == 'France Vs Germany Nuclear Energy':
    query01='''
    select * from  main_db.typesplit2
    where year > 1990 ;
    '''
    query02='''
    SELECT year, country_code,energy_type_code, energy_consumption, co2_emission
    FROM "main_db"."energyco2" 
    where country_code in ('DEU', 'FRA')and year > 1991 ;
    '''
    @st.experimental_memo(ttl=86400)
    def request(query):
        return apirequest(query)

    dfq = request(query01)  ############# llamado api
    dfq2 = request(query02)

    @st.experimental_memo
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

        return fig

    @st.experimental_memo
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

        return fig
        
    st.subheader('Comparison of emissions by type of energy consumed Germany-France')

    st.write(relative_energy_consumption('DEU', 'Germany'))
    st.write(relative_energy_consumption('FRA', 'France'))

    st.markdown('''It is observed how Germany despite 
                   [increasing the production of wind-solar energy](https://www.worldenergytrade.com/politica/europa/germany-increases-its-objetivo-de-energia-limpia-del-65-al-80-by-2030) 
                   has not significantly decreased in consumption of fossil sources (80% in 2019) especially those 
                   that come from coal. On the other hand, France, which maintains a high share of nuclear energy, 
                   manages to proportionally reduce its fossil fuel consumption (<50%)''')
    st.markdown('This is reflected in a large difference in CO2 emissions per unit of energy consumed:')
    st.write(compare())                      
            

    st.markdown('''The explanation is possibly the intermittent nature of solar and wind energy 
                   (they do not produce at night, they depend on the climate, geography and seasons) 
                   which forces them to make up for the shortcoming with the support of conventional 
                   sources, which also increases costs. 
                ''')


    st.subheader('Conclusion')
    st.markdown('''From the above, it is suggested to increase nuclear energy sources to reduce 
                   greenhouse gas emissions.''')
             
elif choose == 'Countries Clusters by relationship between GDP and CO2 Emission':
    @st.experimental_memo
    def plot_cluster(year):
        """Function that plots a bubble plot with Countries classified with clusters,
        generated with sklearn's K-Means, as a result of the relationship between
        GDP and CO2 Emissions

        Args:
            year (Integer): Year to plot

        Returns:
            Plotly bubble plot
        """
        data = pd.read_csv('./Data_cleansing/csv_export_1/energyco2.csv')
        data_2019 = data[data['Year'] == year]
        data_2019 = data_2019[['Country_Code', 'Energy_Consumption', 'Energy_Intensity_Per_Capita', 'Co2_Emission']]

        country_info = pd.read_csv('./Data_cleansing/csv_export_1/country_info.csv')
        country_info_19 = country_info[country_info.Year == year]
        country_info_19 = country_info_19[['Country_Code', 'Gdp']]

        countries = pd.read_csv('./Data_cleansing/csv_export_1/dim_country.csv')
        countries = countries[['Country', 'Country_Code']]

        data_2019 = data_2019.groupby('Country_Code').sum()
        data_merged = data_2019.merge(country_info_19, left_on='Country_Code', right_on='Country_Code')
        data_merged = data_merged.merge(countries, left_on='Country_Code', right_on='Country_Code')
        data_merged.dropna(inplace=True)

        x = data_merged[['Gdp']]
        x_labels = data_merged['Country']

        kmeans = KMeans(n_clusters=3,
                        init='k-means++',
                        random_state=42)
        kmeans_19 = kmeans.fit(x)

        data_merged['Emission_Category_id'] = kmeans_19.labels_
        data_merged['Emission_Category_str'] = data_merged['Emission_Category_id']
        values = {'Emission_Category_str': {0: 'Low', 2:'Medium', 1:'High'}}
        data_merged = data_merged.replace(values)
        data_merged.reset_index(inplace=True, drop=True)

        fig = px.scatter(data_merged, x='Country_Code', y='Co2_Emission', color='Emission_Category_str',
                        size='Co2_Emission', hover_data=['Country', 'Gdp'], 
                        labels={'Emission_Category_str':'Emission Category',
                                'Co2_Emission':'CO2 Emission (Million Tonnes)',
                                'Country_Code':''},
                        title=f'Countries Clusters by relataionship between GDP and CO2_Emission in {year}')
        return fig

    y = st.slider('Select a year', 1990, 2019, 2019)
    st.plotly_chart(plot_cluster(y))
    st.markdown("""
        The size of the bubble represents the amount of CO2 emmited by the country in millions tonnes. The emission category was
        made with K-Means using GDP and CO2 Emission as values to consider and 3 clusters to classify the
        countries. It's also possible to zoom into the plot thanks to plotly.
        
        For 2019, for example, the models classifies US and China as the only two countries in the high emission category
        this happens as China doubles the CO2 emission of the US, and the US doubles the amount of CO2
        emmited by India.
        """, unsafe_allow_html=True)

elif choose == 'Predict the CO2 emission using the share percentage of energy production':
    st.header('Predict the CO2 emission of any country using the share percentage of energy production')
    
    st.markdown("""
    <div style="text-align: justify;">
    This section uses a GradientBoostRegression model to predict the future emision of any country in the dataset
    with available data, using the percentage of participation of each type of fuel in the energy generation
    process.

    Each country has a different error value as the model was fitted for each country to avoid the introduction
    of noise. The error, as is in the same scale as the final value, can be interpreted as the positive or negative
    difference of the value predicted, so for example, if the value predicted is 100 and the error says 50. The
    result can be interpreted as 100 million tonnes of CO2 emitted +/- 50 million tonnes, which in other words means that
    the real emission can be 150 or 50 million tonns.

    To use the model, simply input the percentages in each fuel type, 10% = 10. If the sum of the percentages is bigger
    than 100, the model will show a message asking you to correct the values. If the values are correct, or less than 100, it will show
    the prediction and the increase (in red) or reduction (in green)
    </div>
    """, unsafe_allow_html=True)

    query_share = """
        SELECT * FROM main_db.energy_share
        WHERE year >= 1990;
    """

    df_share = request(query_share)
    df_share = df_share.drop('greenhouse_gas_emissions', axis=1)

    query_emissions = """
        SELECT * FROM main_db.energyco2
        WHERE year >= 1990;
    """
    emission = request(query_emissions)
    emission = emission.groupby(['country_code', 'year']).sum()
    emission.reset_index(inplace=True, drop=False)
    emission = emission[['country_code', 'year', 'co2_emission']]

    data_merged = df_share.merge(emission, left_on=['country_code', 'year'], right_on=['country_code', 'year'])
    data_merged.dropna(inplace=True)

    query_country = """
    SELECT * FROM main_db.dim_country;
    """

    df_country = request(query_country)
    df_country = df_country[['country_code', 'country']]
    data_merged = data_merged.merge(df_country, left_on='country_code', right_on='country_code')

    pais_lst = data_merged.country.unique()

    pais = st.selectbox(
        'Select a country',
        pais_lst
    )
    col1, col2 = st.columns(2)

    with col1:
        st.header(f"Last year: {data_merged['year'].max()}")
        last_year = data_merged[(data_merged['country'] == pais) & (data_merged['year'] == data_merged['year'].max())]

        st.metric('% Biofuel electricity share', last_year['biofuel_share_elec'])
        st.metric('% Coal electricity share', last_year['coal_share_elec'])
        st.metric('% Gas electricity share', last_year['gas_share_elec'])
        st.metric('% Hydroelectric electricity share', last_year['hydro_share_elec'])
        st.metric('% Nuclear electricity share', last_year['nuclear_share_elec'])
        st.metric('% Oil electricity share', last_year['oil_share_elec'])
        st.metric('% Solar electricity share', last_year['solar_share_elec'])
        st.metric('% Wind electricity share', last_year['wind_share_elec'])
        st.metric('Last Year Emission', '{} Million Tonnes CO{}'.format(round(last_year['co2_emission'].values[0], 3), '\u2082'))

    with col2:
        st.header("Values to predict")
        biofuel = st.number_input(
            'Insert the percentage of energy generated by biofuel',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )

        coal = st.number_input(
            'Insert the percentage of energy generated by coal',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )

        gas = st.number_input(
            'Insert the percentage of energy generated by gas',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )

        hydro = st.number_input(
            'Insert the percentage of energy generated by hydroelectric',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )
        
        nuclear = st.number_input(
            'Insert the percentage of energy generated by nuclear sources',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )

        oil = st.number_input(
            'Insert the percentage of energy generated by oil sources',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )

        solar = st.number_input(
            'Insert the percentage of energy generated by solar sources',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )

        wind = st.number_input(
            'Insert the percentage of energy generated by wind sources',
            min_value = 0.0, 
            max_value = 100.0,
            step=0.01
        )

    def predict_co2emission(country, biofuel_share_elec, coal_share_elec,
        gas_share_elec, hydro_share_elec, 
        nuclear_share_elec, oil_share_elec,
        solar_share_elec, wind_share_elec):

        """
        Function that recieves a country and energy share production in
        percentage. Looks for the country in the dataset, trains a 
        Gradient Boost Regressor and predicts the amount of CO2 emitted
        using the values passed

        Returns:
            test_score: Model score after testing
            co2_pred: prediction of CO2 emitted
        """
        
        data = data_merged[data_merged.country == country]
        features_prediction = np.array([biofuel_share_elec, coal_share_elec,
        gas_share_elec, hydro_share_elec,
        nuclear_share_elec, oil_share_elec,
        solar_share_elec, wind_share_elec]).reshape(1, -1)

        x = data[['biofuel_share_elec', 'coal_share_elec',
        'gas_share_elec', 'hydro_share_elec', 
        'nuclear_share_elec', 'oil_share_elec',
        'solar_share_elec', 'wind_share_elec']]

        y = data[['co2_emission']]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=42)
        reg = GradientBoostingRegressor(random_state=42)
        reg.fit(x_train, np.ravel(y_train, ))
        train_pred = reg.predict(x_train)
        train_score = mean_squared_error(y_train, train_pred, squared=False)
        test_pred = reg.predict(x_test)
        test_score = mean_squared_error(y_test, test_pred, squared=False)

        reg.fit(x, y)
        co2_pred = reg.predict(features_prediction)

        return test_score, co2_pred

    st.header('Prediction results')

    total = biofuel + coal + gas + hydro + nuclear + oil + solar + wind
    if total > 100:
        st.markdown('<p style="color:Red; font-size:22px">The total can\'t be over 100, check your values </p>', unsafe_allow_html=True)
    elif total == 0:
        col3, col4 = st.columns(2)
        with col3:
            st.metric('Test error', 0)
        with col4:
            st.metric('Prediction rounded (million tonnes CO{})'.format('\u2082'), 
                    0, 0, delta_color='inverse')
    else:
        test_score, co2_pred = predict_co2emission(pais, biofuel, coal, gas, hydro, nuclear, oil, solar, wind)
        col3, col4 = st.columns(2)
        with col3:
            st.metric('Test error', round(test_score, 3))
        with col4:
            st.metric('Prediction rounded (million tonnes CO{})'.format('\u2082'), 
                    round(co2_pred[0], 3),
                    round(co2_pred[0] - last_year['co2_emission'].values[0], 3),
                    delta_color='inverse')

elif choose == 'Forecasting consumption and emission':
    timestr = time.strftime("%Y%m%d-%H%M%S")
    def csv_downloader(data):
        csvfile = data.to_csv()
        b64 = base64.b64encode(csvfile.encode()).decode()
        new_filename = "new_text_file_{}_.csv".format(timestr)
        st.markdown("#### Download  CSV ###")
        href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">download</a>'
        st.markdown(href,unsafe_allow_html=True)

    class FileDownloader(object):
        """docstring for FileDownloader
        >>> download = FileDownloader(data,filename,file_ext).download()

        """
        def __init__(self, data,filename='dataset',file_ext='txt'):
            super(FileDownloader, self).__init__()
            self.data = data
            self.filename = filename
            self.file_ext = file_ext

        def download(self):
            b64 = base64.b64encode(self.data.encode()).decode()
            new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
            st.markdown("#### Download CSV ###")
            href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">download</a>'
            st.markdown(href,unsafe_allow_html=True)

    query01='''
    select * from  main_db.typesplit2
    where year > 1990 ;
    '''
    query02='''
    SELECT * FROM "main_db"."year_cou_tot_emiss" 
    where year > 1990 ;
    '''
    query03 =''' select * 
    from  "main_db"."country_info";

    '''
    query_country = """
    SELECT * FROM main_db.dim_country;
    """

    df_type = request(query01)  ############# llamado api
    df_emission = request(query02)
    df_cou_info = request(query03)
    df_country_dim = request(query_country)
    df_country_dim = df_country_dim[['country_code', 'country']]

    df_gen= pd.merge(df_cou_info, df_type,  on= ['year', 'country_code']) 
    df_gen = pd.merge(df_gen, df_emission, on= ['year', 'country_code'])

    df_group_year= df_gen.groupby(['year']).sum().reset_index() #totales por año

    rango= np.arange(2020,2031).reshape(-1,1)

    df_country= df_gen.groupby(['year', 'country_code']).sum().reset_index()
    df_country = pd.merge(df_country, df_country_dim, on= ['country_code'])

    list_country= df_country.country.unique()

    color_map = ["#342E37", "#928779", "#34495e", "#DD403A", "#A2D729"]

    def makegraph(country):
        df = df_country.loc[df_country.country == country].reset_index(drop=True)
            
        df= df.drop(['country_code'], axis=1)
        
        X = df.iloc[:,0].values.reshape(-1,1)
        df_pron= pd.DataFrame()
        df_pron[['year']] = rango
        
        
        
        for i in range(1, 9):
        
            y = df.iloc[:, i]
            regr= LinearRegression()
            regr.fit(X, y)
            y_pred = regr.predict(X)
            y_prediction= regr.predict(rango)
            df_pron[[i]] = y_prediction.reshape(-1,1)
        
        df_pron.rename(columns={1:'population', 2:'gdp', 3: '1', 4:'2', 5:'3', 6:'4', 7:'5', 
                                8:'total_emission'}, inplace=True )
            
        df_pron = pd.concat([df, df_pron], axis=0)    
        
        y1=df_pron['1']
        y2=df_pron['2']
        y3=df_pron['3']
        y4=df_pron['4']
        y5=df_pron['5']
        
        
        fig, ax = plt.subplots(figsize= (12, 6), dpi=120)
        
        ax.set_xlabel('year',   fontsize=16)
        ax.set_ylabel(' consuption in quad btu', fontsize=16)
        ax.set_title('Forecast emission & energy consumption by type, country: '+ country , fontsize=18)
        
        ax.stackplot(df_pron.year,  y1 , y2, y4, y3, y5, labels=['coal ','natural gas', 'petroleum and other liquids',
                    'nuclear', 'renewables and others'], colors=color_map )
        
        ax2 = ax.twinx()
        ax2.axvline(x=2020, color='blue', label='Forecast 2020-2030')
        ax2.axvline(x=2005, color='brown', ls='--', label='Kioto protocol')
        ax2.axvline(x=2016, color='brown', ls='-.', label='Paris agreement')
        
        
        ax2.plot(df_pron.year, df_pron.total_emission,'--', linewidth=2, color='black',label='total emission co2' )
        ax2.set_ylabel('co2 emission in megaton', fontsize=16)
        
        ax2.set_ylim(0)
        ax.grid(True)
        
        ax.legend(loc='lower left',prop={'size': 12}, bbox_to_anchor=(1.1, 0.5));
        ax2.legend(loc='upper left', prop={'size': 12}, bbox_to_anchor=(1.1, 0.5))
        return fig

    st.title('Forecasting energy consumption and CO2 emission ')
    st.subheader('simple linear regression model ')

    country= st.multiselect('Select countries', list_country, default=['Argentina'])   #####  menu STREAMLIT
    for i in range(len(country)):
        item = country[i]   
        st.write(makegraph(item))
        
    df_country.rename(columns= {'1': 'coal', '2':'natural gas', '3': 'nuclear', '4':'petroleum other liquids',
                                '5': 'renewables and others'}, inplace=True)   

    download = FileDownloader(df_country.to_csv(),file_ext='csv').download()

elif choose == 'CO2 prediction with Neural Network':    
    
    @st.experimental_memo(ttl=86400)
    def request(query):
        return apirequest(query)

    st.title('3d Map / Prediction of Emission CO2')

    st.subheader('The purple bars show, the amount of CO2 emitted between 1980 and 2019 by Country')
    st.subheader('How to use')
    st.caption('This is a map in 3d, so you can move it with 2nd mouse button.')
    st.caption('Mouse wheel, zoom in or zoom out.')

    #Se carga el modelo entrenado
    modelo_final = joblib.load('./modelo_entrenado.pkl')
    #Query desde AWS
    query =''' SELECT * FROM "db_clean3"."energyco2_origin" 
    where not country ='World' and not energy_type ='all_energy_types;'''
    #df_co2_completo = request(query) #me da error 502 (bad gateway)

    #solucion para la prueba ingestando desde el dataset
    df_co2_completo = pd.read_csv('./Datasets/energyco2.csv')#Es el dataset original
    df_countries = pd.read_csv('./Datasets/dim_country.csv')
    df_energy_clean = pd.read_csv('./Data_cleansing/csv_export_1/energyco2.csv')#es el dataset que limpió Aurora y Ezequiel
    #trabajando con los datasets
    df_countries.rename(columns={'Latitude(average)':'latitude','Longitude(average)':'longitude'},inplace=True)
    df_co2_clean = df_energy_clean.drop(['Energy_Type_Code','Energy_Consumption','Energy_Intensity_Per_Capita','Energy_Intensity_By_Gdp','Year'], axis=1)
    CO2_Sum = df_co2_clean.groupby(['Country_Code']).sum().reset_index()
    df_co2_e = pd.DataFrame(CO2_Sum)
    df_co2_e.rename(columns={df_co2_e.columns[0]:'Country_Code'},inplace=True)
    countries_geo = df_countries.iloc[:, [2,3]]
    countries_joined = df_countries.merge(df_co2_e, on='Country_Code').reset_index()
    countries_joined.dropna(inplace=True)
    #configuración inicial del mapa
    midpoint=[np.average(countries_joined["latitude"]),np.average(countries_joined["longitude"])]
    #st.write("{:.4f}".format(midpoint[0]),midpoint[1])
    st.pydeck_chart(pdk.Deck(
        map_style= 'mapbox://styles/mapbox/light-v9',
        initial_view_state={
            #"{:.2f}".format(z)
            'latitude': midpoint[0],
            'longitude': midpoint[1],
            'zoom':0,
            'pitch':30
            }, 
            layers=[
                pdk.Layer(
                    "ColumnLayer",
                    data= countries_joined,                
                    get_position= ["longitude","latitude"],
                    get_elevation=['Co2_Emission'],
                    radius=200000,
                    get_fill_color=[180, 0, 200, 140],
                    elevation_scale=100,
                    elevation_range=[0,2000],
                    pickable= True,
                    extruded= True,
                )
            ]
    ))


    #st.write(countries_joined)
    #st.write(df_co2_e)
    mask_AET = df_co2_completo['Energy_type']!='all_energy_types'
    df_co2_AET = df_co2_completo[mask_AET]
    mask_world = df_co2_AET['Country'] != 'World'
    df_co2_final = df_co2_AET[mask_world]
    #df_co2_final = df_co2_completo
    #st.write(df_co2_final)
    #este feature se puede eliminar, pero yo le agarré cariño para las pruebas
    df_co2_final.rename(columns={"Unnamed: 0": 'Index'},inplace=True)

    #aquí year_sel y user_sel vienen de una lista desplegable
    country_list = df_co2_final['Country'].unique()
    #st.write(country_list)

    st.subheader('The simulation shows the amount of CO2 emitted, according to the values entered by the user.')
    st.subheader('How to use')
    st.caption('This is a simulation that allows you to see the impact on carbon emissions, according to the use of a certain type of energy, consumption, production, inhabitants, GDP.')
    st.caption('All features have an effect on the result, but as you will see some have a more significant effect than others.')
    year_list = df_co2_final['Year'].unique()
    year_sel = st.selectbox(label='Select a Year', options=year_list)
    st.caption('If the table has values in the "CO2_Emission" column, try to change the year, otherwise the emission will be simulated, for that data.')
    Country_sel = st.selectbox(label='Select a Country', options=country_list)
    user_sel = df_co2_final[(df_co2_final.Country == Country_sel) & (df_co2_final.Year == year_sel)].reset_index(drop=True)
    energy_list = df_co2_final['Energy_type'].unique()

    #Aquí se separan los valores para después probarlos
    user_sel.fillna(0.0,inplace=True)
    #df_ver_test = user_sel['CO2_emission']


    #Se muestra el dataset resultante para información del usuario

    # CSS to inject contained in a string
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.caption('If the table is not fully visible, Please check the scroll bar at the bottom of it.')
    st.table(user_sel[['Energy_type','Energy_consumption','Energy_production','Population','GDP','Energy_intensity_per_capita','Energy_intensity_by_GDP','CO2_emission']])
    #st.plotly_chart(user_sel)

    #Se toma una sola fila para modificarla
    X_new = user_sel.head(1)

    #Donde podemos cambiar los datos

    X_new['Energy_type'] = st.selectbox(label='Select a Energy type', options=energy_list)
    E_type = X_new['Energy_type'][0]
    index_Energy = user_sel.index[user_sel['Energy_type'] == E_type].values
    X_new['Year'] = year_sel
    X_new['Energy_consumption'] = st.number_input('Insert a Energy Consumption number',step=1.00,format='%.5f',value=user_sel.iloc[index_Energy[0],4])
    X_new['Energy_production'] = st.number_input('Insert a Energy Production number',format='%.5f',value=user_sel.iloc[index_Energy[0],5])
    X_new['Population'] = st.number_input('Insert a Population number',step=1000.00,value=user_sel.iloc[index_Energy[0],7])
    X_new['GDP'] = st.number_input('Insert a GDP number',step=20.00,format='%.2f',value=user_sel.iloc[index_Energy[0],6])
    X_new['Energy_intensity_per_capita'] = st.number_input('Insert a Energy Intensity per capita number',step=1.0,value=user_sel.iloc[index_Energy[0],8])
    X_new['Energy_intensity_by_GDP'] = st.number_input('Insert a Energy Intensity by GDP number',step=1.0,value=user_sel.iloc[index_Energy[0],9])

    #Uso del modelo entrenado
    predicciones_final = modelo_final.predict(X = X_new)

    co2_original = user_sel.iloc[index_Energy[0],-1]

    result = float(predicciones_final)- float(co2_original)
    st.metric(label="Value of Emission of CO2 (millons of Tons) - Predicted", value= np.round(predicciones_final,4),delta=np.round(result,4))
    st.caption('As you could check. The reduction in energy consumption, the use of clean energy reduces the impact on carbon emissions.')
    st.caption('This is just an exercise for educational purposes.')

elif choose == 'CO2 emissions (approximation)':
    st.write('# CO2 emissions (approximation)')
    st.write('''
            The following prediction, models four indicators: human population, GDP per capita, energy intensity (per unit of GDP), and carbon intensity (emissions per unit of energy consumed) and approximates CO2 emissions by country based on these factors. 
            Only countries with less than 50% of missing values are presented. 
            ''')
    country2 = pd.DataFrame(country1)
    country2.rename(columns={0:'Country_Code'}, inplace=True)
    country2 = pd.merge(country2, dim_country, on='Country_Code', how='left')
    country_name = tuple(country2['Country'].values)
    years = []
    for i in range(2023, 2031):
        years.append(str(i))

    col1_1, col1_2, col1_3 = st.columns([1,1,2])
    with col1_1:
        c = st.selectbox('Select country', country_name)
    with col1_2:
        y = st.selectbox('Select year', years)

    col2_1, col2_2 = st.columns(2)
    cn = country2[country2['Country'] == c].index.values[0]
    c_code = country2.loc[cn, 'Country_Code']

    df_c = df_country(c_code)
    predict(c_code, 2030)
    idx = list(pred[pred['pred_co2']<0].index.values)
    pred.loc[idx, 'pred_co2'] = 0

    col2_1.write('##### Graph prediction(factors of environmental impact)')
    col2_1.write(plot_pred(df_c, coeficientes))
    col2_2.write(f'##### Prediction for the year {y}')
    index = pred[pred['Year'] == int(y)].index.values[0]
    co2 = pred.loc[index,'pred_co2']
    col2_2.write(f'CO2 emissions: {round(co2, 2)} millions of tons')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.arange(1980, 2031), y=df_c['Co2_Emission'], mode='markers', marker=dict(color='#004600'), name='CO2 emission'))
    fig.add_trace(go.Scatter(x = np.arange(2020, int(y)+1), y=pred['pred_co2'], mode='markers', marker=dict(color='#707070'), name='CO2 prediction'))
    fig.update_layout(title='CO2 emission',
    xaxis=dict(title='Year', gridcolor='#E2E2E2', griddash='dash', ticks='outside', tickcolor='#000000'),
    yaxis=dict(title='CO2 emission(millions of tons)', gridcolor='#E2E2E2',griddash='dash', ticks='outside', tickcolor='#000000'), plot_bgcolor='rgba(0,0,0,0)')
    col2_2.write(fig)

if choose == 'Predict the emission of a house in a country':

    st.header('Predict the emission of a house in a country')
    st.markdown('''
    This is a KNN model, in which you put a coordinate in the model and it will returns the CO2 emissions that this 
    house could produce, this is achieved by taking out the 5 closest power plants to the house and from most 
    of these, obtains the type of energy to do the previous calculation''')
    
    #m = folium.Map(location=[46.1991, -122.1889], tiles="Stamen Terrain", zoom_start=13)
    #m.add_child(folium.LatLngPopup())
    #m
 
    # Se llama a los df
    plant_info = pd.read_csv(
        './Data_cleansing/csv_export/plant_info.csv')
    plant_generation = pd.read_csv(
        './Data_cleansing/csv_export/plant_generation.csv')
    dim_fuel = pd.read_csv(
        './Data_cleansing/csv_export/dim_fuel.csv')
    energyco2 = pd.read_csv(
        './Data_cleansing/csv_export/energyco2.csv')
    country_info = pd.read_csv(
        './Data_cleansing/csv_export/country_info.csv')
    
    st.title("Test")
    countrylist = plant_info['Country_Code'].unique()
    option = st.selectbox('Choose a country to see power plants',options=countrylist,index=0)
    plant1 = plant_info[plant_info['Country_Code']==option]
    m = folium.Map((15.284185,129.37500), tiles="Stamen Terrain", zoom_start=1)
    #folium.Marker([46.8354, -121.7325], popup="Camp Muir").add_to(m)
    #for i in plant_info.index: 
    #print("Total income in "+ df["Date"][i]+ " is:"+str(df["Income_1"][i]+df["Income_2"][i]))
    #    folium.Marker([plant_info['Latitude'][i], plant_info['Longitude'][i]], popup=plant_info['Name'][i]).add_to(m)
    plant1.apply(lambda row:folium.CircleMarker(location=[row['Latitude'], row['Longitude']], 
                                              radius=1, popup=row['Name'])
                                             .add_to(m), axis=1)
    m.add_child(folium.ClickForMarker(popup="Markdown"))
    #m.save("map test.html", close_file=True)
    #st.markdown(m._repr_html_(), unsafe_allow_html=True)
    map = st_folium(m, width = 1000, height=500)

    # Se genera un nuevos df donde se tenga todos los datos necesarios
    
    plant_info2 = dim_fuel.join(
        plant_generation.set_index('Fuel_Code'), on='Fuel_Code')
    plant_info2 = plant_info2[['Fuel', 'Name']]
    plant_info2 = plant_info2.join(plant_info.set_index('Name'), on='Name')
    plant_info2.reset_index(inplace=True, drop=True)

    country_info2 = country_info[country_info.Year == 2019]
    country_info2.drop(['Year'], axis=1, inplace=True)

    energyco1 = energyco2[energyco2.Year == 2019]
    energyco1 = energyco1.join(dim_fuel.set_index(
        'Fuel_Code'), on='Energy_Type_Code')
    energyco1 = energyco1.join(country_info2.set_index(
        'Country_Code'), on='Country_Code')
    energyco1 = energyco1[['Country_Code', 'Fuel', 'Co2_Emission', 'Population']]
    energyco1.reset_index(inplace=True, drop=True)

    # inputs para ingresar latitud y longitud
    #lat = float(input('Ingrese latitud'))
    #lon = float(input('Ingrese longitud'))
    #map['last_clicked']['lat']
    latitude1 = map['last_clicked']['lat']
    longitude1 = map['last_clicked']['lng']
    st.write(latitude1)
    st.write(longitude1)
    lat = st.number_input('Latitude',value=latitude1)
    lon = st.number_input('Longitude', value=longitude1)

    # se hace un knn el cual trae el tipo de energia
    #lat, lon = 40.657319, -75.646606
    data = plant_info2[['Latitude', 'Longitude']]
    plant = plant_info2.Fuel

    es = preprocessing.MinMaxScaler()
    data = es.fit_transform(data)

    clas = KNeighborsClassifier(n_neighbors=5)
    clas.fit(data, plant)
    plantClose = es.transform([[lat, lon]])
    fuel = clas.predict(plantClose)[0]

    # se hace un knn el cual trae el pais
    data = plant_info2[['Latitude', 'Longitude']]
    plant = plant_info2.Name

    data = es.fit_transform(data)

    clas.fit(data, plant)

    plantClose = es.transform([[lat, lon]])
    country = plant_info2[plant_info2.Name == clas.predict(plantClose)[0]]
    country = country.Country_Code.values[0]

    # se calcula las emisiones de co2 para esa supuesta casa
    house_emission = energyco1[(energyco1['Country_Code'] == country) & (
        energyco1['Fuel'] == fuel)]
    house_emission = float(
        (house_emission.Co2_Emission.values/house_emission.Population.values) * 4)

    col111, col222 = st.columns(2)

    with col111:    
        st.metric('Emisiones CO2 por casa', house_emission)
    with col222:
        st.metric('Tipo de energía que provee a la vivienda', fuel)