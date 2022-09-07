import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
from apiathena import apirequest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu

@st.experimental_memo(ttl=86400)
def request(query):
    return apirequest(query)

with st.sidebar:
    choose = option_menu("Insights & Predictions", ['France Vs Germany Nuclear Energy',
                                  'Countries Clusters by relationship between GDP and CO2 Emission',
                                  'Predict the CO2 emission of any country using the share percentage of energy production',
                                  'Forecasting consumption and emission'])


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
        
        For 2019 the models classifies US and China as the only two countries in the high emission category
        this happens as China doubles the CO2 emission of the US, and the US doubles the amount of CO2
        emmited by India.
        """)

elif choose == 'Predict the CO2 emission of any country using the share percentage of energy production':
    st.header('Predict the CO2 emission of any country using the share percentage of energy production')
    

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
        
        data = data_merged[data_merged.country_code == country]
        features_prediction = np.array([biofuel_share_elec, coal_share_elec,
        gas_share_elec, hydro_share_elec,
        nuclear_share_elec, oil_share_elec,
        solar_share_elec, wind_share_elec]).reshape(1, -1)

        x = data[['biofuel_share_elec', 'coal_share_elec',
        'gas_share_elec', 'hydro_share_elec', 
        'nuclear_share_elec', 'oil_share_elec',
        'solar_share_elec', 'wind_share_elec']]

        y = data[['co2_emission']]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
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

    country_name='Brasil'

    list_country= df_country.country.unique()

    country= st.selectbox('choice country', list_country)

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
                'nuclear', 'renewables and others'])
    ax.axvline(x=2019, color='black', label='Forecast 2020-2030')
    ax2 = ax.twinx()
    ax2.plot(df_pron.year, df_pron.total_emission,'--', linewidth=2, color='black',label='total emission co2' )
    ax2.set_ylabel('co2 emission in megaton', fontsize=16)

    ax2.set_ylim(0)
    ax.grid(True)

    ax.legend(loc='lower left',prop={'size': 12});
    ax2.legend(loc='upper left', prop={'size': 12})
    st.pyplot(fig)