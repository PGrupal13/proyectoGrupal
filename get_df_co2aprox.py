from apiathena import apirequest 
import string as s
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Auxiliar functions
def capword(list_values, sep = '_'):
    '''
    Capitalize the first letter of each word in each element of list
    list_values: list of strings
    sep: separator, default = '_'
    '''
    for i in range(len(list_values)):
        list_values[i] = list_values[i].lower()
        list_values[i] = s.capwords(list_values[i], sep=sep)
    return list_values


def cap_namecolumns(df, sep = '_'):
    '''
    Capitalize the first letter of each word in dataframe column name
    df: Dataframe 
    sep: Separator, default = '_'
    '''
    col_old = list(df.columns.values)
    col_new = capword(list(df.columns.values), sep=sep)
    columns_name = dict(zip(col_old, col_new))
    df.rename(columns = columns_name, inplace=True)


q1 = 'SELECT * FROM main_db.country_info'
q2 = 'SELECT * FROM main_db.dim_country'
q3 = 'SELECT * FROM main_db.energyco2_all'

country_info = apirequest(q1)
dim_country = apirequest(q2)
energyco2 = apirequest(q3)
cap_namecolumns(country_info)
cap_namecolumns(dim_country)
cap_namecolumns(energyco2)
df_pibpercapita = pd.read_csv('https://raw.githubusercontent.com/PGrupal13/proyectoGrupal/main/Datasets/GDP.PCAP.csv', skiprows=4)
df_pibpercapita.rename(columns={'Country Code':'Country_Code'}, inplace=True)

#basic data
country_info1 = country_info
country_info1 = pd.merge(country_info1, df_pibpercapita, on='Country_Code', how='left')
for i in range(country_info1.shape[0]):
    year = country_info1.loc[i, 'Year']
    country_info1.loc[i, 'Gdp_Capita'] = country_info1.loc[i, str(year)]/1000000000
country_info1 = country_info1.loc[:,['Country_Code', 'Year', 'Population', 'Gdp', 'Gdp_Capita']]
country_info1['Gdp_1'] = country_info1.apply(lambda r: (r['Gdp_Capita']*r['Population']), axis=1)

df_energy = pd.merge(country_info1, energyco2, on=['Country_Code', 'Year'], how='left')
df_energy = df_energy.loc[:,['Country_Code', 'Year', 'Population', 'Gdp', 'Gdp_1', 'Gdp_Capita', 'Energy_Type_Code', 'Energy_Consumption', 'Energy_Intensity_By_Gdp', 'Co2_Emission']]
df_energy['Population'] = df_energy['Population'].apply(lambda x: x/1000)
df_energy['E_I'] = df_energy.apply(lambda r: r['Energy_Consumption']*1000/r['Gdp'], axis=1)
df_energy['E_I2'] = df_energy.apply(lambda r: r['Energy_Consumption']*1000/(r['Gdp_1']), axis=1)
df_energy['I_Co2'] = df_energy.apply(lambda r: r['Co2_Emission']/r['Energy_Consumption'] if r['Energy_Consumption'] != 0 else 0, axis=1)


#Pred
def df_country(country_code):
    df = df_energy[(df_energy['Country_Code'] == country_code) & (df_energy['Energy_Type_Code'] == 0)]
    return df


def predict(country_code, year):
    df_c = df_country(country_code)
    target = ['Population', 'Gdp_Capita', 'E_I2', 'I_Co2']
    df_c = df_c.interpolate(method='linear', limit_direction='both')
    xtrain = np.arange(df_c.Year.min(), df_c.Year.max()+1)
    xtrain = xtrain.reshape(-1, 1)
    xpred = np.arange(df_c.Year.max()+1,year+1)
    xpred = xpred.reshape(-1, 1)
    pred['Year'] = list(xpred.flatten())

    #Linear(Population)
    # for i in range(2):
    ytrain = df_c[df_c['Year'].isin(list(xtrain.flatten()))][target[0]].values
    #Modelo
    m_linear = LinearRegression(fit_intercept=True)
    #Ajuste
    m_linear.fit(xtrain, ytrain)
    #Predicción
    yp = m_linear.predict(xpred)
    #Obtención de datos
    pred['pred_'+target[0]] = pd.Series(yp)
    coeficientes['lineal_'+target[0]] = []
    coeficientes['lineal_'+target[0]].append(m_linear.coef_[0])
    coeficientes['lineal_'+target[0]].append(m_linear.intercept_)

    #Exp(Gdp_Capita, E_I2)
    for i in range(1,3):
        ytrain = df_c[df_c['Year'].isin(list(xtrain.flatten()))][target[i]].values
        #Modelo y Ajuste
        m_expo = np.polyfit(xtrain.flatten(),np.log(ytrain), 1)
        #Predicción
        yp = (np.exp(m_expo[0])**(xpred.flatten()))*np.exp(m_expo[1])
        #Obtención de datos
        pred['pred_'+target[i]] = pd.Series(yp)
        coeficientes['expo_'+target[i]] = []
        coeficientes['expo_'+target[i]].append(np.exp(m_expo[0]))
        coeficientes['expo_'+target[i]].append(np.exp(m_expo[1]))

    #Polynomial(I_Co2)
    ytrain = df_c[df_c['Year'].isin(list(xtrain.flatten()))][target[3]].values
    #Modelo
    m_poly = PolynomialFeatures(degree = 2)
    m_linear = LinearRegression(fit_intercept=True)
    #Ajuste
    xtrain1 = m_poly.fit_transform(xtrain)
    m_linear.fit(xtrain1, ytrain) 
    #Predicción
    yp = (m_linear.coef_[0]*(xpred.flatten()**0)) + (m_linear.coef_[1]*(xpred.flatten()**1))+ (m_linear.coef_[2]*(xpred.flatten()**2))+m_linear.intercept_
    #Obtención de datos
    pred['pred_'+target[3]] = yp
    coeficientes['poly_'+target[3]] = list(m_linear.coef_)
    coeficientes['poly_'+target[3]].append(m_linear.intercept_)

    #Predicción
    pred['pred_co2'] = pred.apply(lambda r: r['pred_I_Co2']*r['pred_E_I2']*r['pred_Gdp_Capita']*r['pred_Population'], axis=1)

    return pred['pred_co2'].values


def plot_pred(df, coef):
    x = df['Year']
    y1 = df['Population']
    y2 = df['Gdp_Capita']
    y3 = df['E_I2']
    y4 = df['I_Co2']
    fig, ax = plt.subplots(4,1,sharex=True, figsize=(10,10))
    ax[0].scatter(x, y1, color='#004600')
    y1_1 = (coef['lineal_Population'][0]*x)+coef['lineal_Population'][1]
    ax[0].plot(x, y1_1, color='#707070', lw=3)
    ax[0].set(title='Population', ylabel='Population')

    ax[1].scatter(x, y2, color='#004600')
    y2_2 = (coef['expo_Gdp_Capita'][0]**x)*coef['expo_Gdp_Capita'][1]
    ax[1].plot(x, y2_2, color='#707070', lw=3)
    ax[1].set(title='GDP per cápita', ylabel='GDP per cápita')

    ax[2].scatter(x, y3, color='#004600')
    y3_3 = (coef['expo_E_I2'][0]**x)*coef['expo_E_I2'][1]
    ax[2].plot(x, y3_3, color='#707070', lw=3)
    ax[2].set(title='Energy intensity of the GDP', ylabel='Energy intensity(1000 Btu/GDP)')

    ax[3].scatter(x, y4, color='#004600')
    y4_4 = (coef['poly_I_Co2'][0]*(x**0))+(coef['poly_I_Co2'][1]*(x**1))+(coef['poly_I_Co2'][2]*(x**2))+coef['poly_I_Co2'][3]
    ax[3].plot(x, y4_4, color='#707070', lw=3)
    ax[3].set(title='Carbon footprint of energy', xlabel='Year', ylabel='Carbon footprint of energy')

    return fig

pred = pd.DataFrame()
coeficientes = {}

country = list(df_energy['Country_Code'].unique())
country1 = []
for i in country:
    df_c = df_country(i)
    if df_c['Population'].isna().sum() > df_c.shape[0]*.4 or df_c['Gdp_Capita'].isna().sum() > df_c.shape[0]*.4 or df_c['E_I2'].isna().sum() > df_c.shape[0]*.4 or df_c['I_Co2'].isna().sum() > df_c.shape[0]*.4:
        continue
    else:
        country1.append(i)




