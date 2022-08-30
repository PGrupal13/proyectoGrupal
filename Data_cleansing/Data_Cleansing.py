import numpy as np
import pandas as pd
import string as s
from sklearn import preprocessing
import requests
from zipfile import ZipFile
from io import BytesIO

#Datasets
df_energy_consumption = pd.read_csv('../Datasets/owid-energy-consumption-source.csv')
df_energyco2 = pd.read_csv('../Datasets/energyco2.csv')
df_power_plant = pd.read_csv('../Datasets/global_power_plant_database.csv')
dim_country = pd.read_csv('../Datasets/dim_country.csv')


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


def rename_country(df, list_country_old, list_country_new):
    '''
    Rename countries in dataframe
    list_country_old: Current list of country names
    list_country_new: List of new country names
    '''
    for i in range(len(list_country_old)):
        index = df[df['Country'] == list_country_old[i]].index.values
        df.loc[index, 'Country'] = list_country_new[i]


def remote_file(url, file_name):
    '''
    Return dataframe from a csv in remote zip file
    url: url file
    file_name: name of csv file
    '''
    fget = requests.get(url)
    fzip = ZipFile(BytesIO(fget.content))   
    with fzip.open(file_name) as file:
        df = pd.read_csv(file, skiprows=4)
    return df

#Rename columns
cap_namecolumns(df_energy_consumption)
cap_namecolumns(df_energyco2)
cap_namecolumns(df_power_plant)
dim_country.rename(columns={'Latitude (average)': 'Latitude', 'Longitude (average)': 'Longitude'}, inplace=True)
df_power_plant.rename(columns={'Country': 'Country_Code', 'Country_Long': 'Country'}, inplace=True)
#Remove blanks
dim_country.loc[:, ['Country', 'Country_Code']] = dim_country.loc[:, ['Country', 'Country_Code']].applymap(lambda x: x.strip())
df_power_plant['Country_Code'] = df_power_plant['Country_Code'].apply(lambda x: x.strip())
#-----------------------------------------------------
#Add teritories
territories = ['The Union of Soviet Socialist Republics', 'Serbia and Montenegro', 'Yugoslavia', 'Czechoslovakia']
codes = ['SUN', 'SCG', 'YUG', 'CSK']
latitud = [61.52401, 44.6583, 44.818996724, 50.073658]
longitud = [105.318756, 20.6844, 20.457331504, 14.418540]
for i in range(4):
    row = dim_country.shape[0]
    dim_country.loc[row, 'Country']= territories[i]
    dim_country.loc[row, 'Country_Code']= codes[i]
    dim_country.loc[row, 'Latitude']= latitud[i]
    dim_country.loc[row, 'Longitude']= longitud[i]
dim_country.sort_values('Country', inplace=True)
dim_country.reset_index(inplace=True, drop=True)
#-------------------------------------------------------

#Rename countries
#df_energyco2



l1 = ['The Bahamas', 'British Virgin Islands', 'Cabo Verde', 'Congo-Brazzaville',
      'Congo-Kinshasa', 'Côte d’Ivoire', 'Falkland Islands', 'Gambia,T he', 
      'Iran', 'North Korea', 'Laos', 
      'Macau', 'Moldova', 'Micronesia', 'Palestinian Territories', 
      'Reunion', 'Saint Helena', 'Saint Vincent/Grenadines', 
      'Eswatini', 'Syria', 'Tanzania', 'U.S. Virgin Islands', 
      'North Macedonia', 'Former Czechoslovakia', 'Former Serbia and Montenegro', 
      'Former U.S.S.R.', 'Former Yugoslavia']

l2 = ['Bahamas', 'Virgin Islands, British', 'Cape Verde', 'Congo', 
      'Congo, the Democratic Republic of the', "Cote d'Ivoire", 'Falkland Islands (Malvinas)', 'Gambia', 
      'Iran, Islamic Republic of', "Korea, Democratic People's Republic of", "Lao People's Democratic Republic", 
      'Macao', 'Moldova, Republic of', 'Micronesia, Federated States of', 'Palestinian Territory, Occupied', 
      'Réunion', 'Saint Helena,  Ascension and Tristan da Cunha', 'Saint Vincent and the Grenadines', 
      'Swaziland', 'Syrian Arab Republic', 'Tanzania, United Republic of', 'Virgin Islands, U.S.', 
      'Macedonia, Republic of North', 'Czechoslovakia', 'Serbia and Montenegro', 
      'The Union of Soviet Socialist Republics', 'Yugoslavia']

rename_country(df_energyco2, l1, l2)

#df_energy_consumption
l3 = ['Czechia', 'Democratic Republic of Congo', 'Falkland Islands', 
      'Faeroe Islands', 'Iran', 'North Korea', 
      'Laos', 'North Macedonia', 'Micronesia (country)', 
      'Moldova', 'Palestine', 'Reunion', 'Eswatini',
      'Syria', 'Tanzania', 'Timor', 'United States Virgin Islands',
      'British Virgin Islands', 'Saint Helena', 'USSR']
    
l4 = ['Czech Republic', 'Congo, the Democratic Republic of the', 'Falkland Islands (Malvinas)',
      'Faroe Islands', 'Iran, Islamic Republic of',  "Korea, Democratic People's Republic of",
      "Lao People's Democratic Republic", 'Macedonia, Republic of North', 'Micronesia, Federated States of', 
      'Moldova,  Republic of', 'Palestinian Territory, Occupied', 'Réunion', 'Swaziland', 
      'Syrian Arab Republic', 'Tanzania, United Republic of', 'Timor-Leste', 'Virgin Islands, U.S.',
      'Virgin Islands, British', 'Saint Helena, Ascension and Tristan da Cunha', 
      'The Union of Soviet Socialist Republics']

rename_country(df_energy_consumption, l3, l4)

#Country filter
df_energy_consumption = df_energy_consumption[df_energy_consumption['Country'].isin(dim_country['Country'].values)]
df_energyco2 = df_energyco2[df_energyco2['Country'].isin(dim_country['Country'].values)]
df_power_plant = df_power_plant[df_power_plant['Country_Code'].isin(dim_country['Country_Code'].values)]


#Categories
le = preprocessing.LabelEncoder()
#Energy(dim)
df_energyco2['Energy_Type_Code'] = le.fit_transform(df_energyco2['Energy_Type'])
dim_energy = pd.DataFrame(le.classes_)
dim_energy.reset_index(inplace=True)
dim_energy.rename(columns = {'index': 'Energy_Type_Code', 0: 'Energy_Type'}, inplace=True)
dim_energy['Energy_Type'] = dim_energy['Energy_Type'].apply(lambda x: x.replace('_', ' ').capitalize())
#Fuel(dim)
df_power_plant['Fuel_Code'] = le.fit_transform(df_power_plant['Primary_Fuel'])
dim_fuel = pd.DataFrame(le.classes_)
dim_fuel.reset_index(inplace=True)
dim_fuel.rename(columns = {'index': 'Fuel_Code', 0: 'Fuel'}, inplace=True)
#Year(dim)
dim_calendar_year = pd.DataFrame(np.arange(df_energy_consumption.Year.min(), 2020))
dim_calendar_year.rename(columns={0: 'Year'}, inplace=True)

#Merge country code
df_energy_consumption = pd.merge(df_energy_consumption, dim_country, on='Country', how='left')
df_energyco2 = pd.merge(df_energyco2, dim_country, on='Country', how='left')

#Energy Co2
columns_energyco2 = ['Country_Code', 'Energy_Type_Code', 'Year', 'Energy_Consumption', 'Energy_Intensity_Per_Capita',
                     'Energy_Intensity_By_Gdp', 'Co2_Emission']
df_energy_co2 = df_energyco2.loc[:, columns_energyco2]

#Energy share
columns_energy_share = ['Country_Code', 'Year', 'Biofuel_Share_Elec', 'Coal_Share_Elec', 'Fossil_Share_Elec',
                'Gas_Share_Elec', 'Hydro_Share_Elec', 'Low_Carbon_Share_Elec', 'Nuclear_Share_Elec',
                'Oil_Share_Elec', 'Other_Renewables_Share_Elec', 'Renewables_Share_Elec', 'Solar_Share_Elec', 
                'Wind_Share_Elec', 'Greenhouse_Gas_Emissions']
df_energy_share = df_energy_consumption.loc[:, columns_energy_share]

#Energy generation
columns_energy_generation = ['Country_Code', 'Year', 'Biofuel_Electricity', 'Coal_Electricity', 'Fossil_Electricity',
                     'Gas_Electricity', 'Hydro_Electricity', 'Nuclear_Electricity', 'Low_Carbon_Electricity',
                     'Oil_Electricity', 'Other_Renewable_Electricity', 'Renewables_Electricity', 'Solar_Electricity',
                     'Wind_Electricity']
df_energy_generation = df_energy_consumption.loc[:, columns_energy_generation]

#Energy consumption
columns_energy_consumption = ['Country_Code', 'Year', 'Biofuel_Consumption', 'Coal_Consumption', 'Fossil_Fuel_Consumption', 'Gas_Consumption',
                      'Hydro_Consumption', 'Nuclear_Consumption', 'Low_Carbon_Consumption', 'Oil_Consumption',
                      'Other_Renewable_Consumption', 'Renewables_Consumption', 'Solar_Consumption', 'Wind_Consumption']
df_energy_consum = df_energy_consumption.loc[:, columns_energy_consumption]

#Plant generation
columns_plant_generation = ['Country_Code', 'Name', 'Fuel_Code', 'Generation_Gwh_2013','Generation_Gwh_2014', 
                           'Generation_Gwh_2015', 'Generation_Gwh_2016', 'Generation_Gwh_2017', 
                           'Estimated_Generation_Gwh_2013', 'Estimated_Generation_Gwh_2014', 'Estimated_Generation_Gwh_2015',
                           'Estimated_Generation_Gwh_2016', 'Estimated_Generation_Gwh_2017','Estimated_Generation_Note_2013', 
                           'Estimated_Generation_Note_2014','Estimated_Generation_Note_2015', 'Estimated_Generation_Note_2016', 
                           'Estimated_Generation_Note_2017']
df_plant_generation = df_power_plant.loc[:, columns_plant_generation]

#Plant info
df_plant_info = df_power_plant.loc[:, ['Country_Code', 'Name', 'Capacity_Mw', 'Latitude', 'Longitude']]

#Country info
df_country_info = df_energyco2.loc[:, ['Country_Code', 'Year', 'Gdp', 'Population']]
df_country_info.drop_duplicates(subset=['Country_Code', 'Year'], inplace=True)

#Null values
df_energyco2.fillna(np.nan, inplace=True)
df_energy_share.fillna(np.nan, inplace=True)
df_energy_generation.fillna(np.nan, inplace=True)
df_energy_consumption.fillna(np.nan, inplace=True)
#Country info
#get remote csv 
df_pop = remote_file('https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv', 
                     'API_SP.POP.TOTL_DS2_en_csv_v2_4413579.csv')
df_pop.rename(columns={'Country Code': 'Country_Code'}, inplace=True)
#years
years = list(df_country_info['Year'].unique())
years.pop()
#Country info(Data for population null values)
aux_info_pop = pd.merge(df_country_info, df_pop, on='Country_Code', how='left')

for i in years:
    index_pop = list(aux_info_pop[aux_info_pop['Year'] == i].index.values)
    aux_info_pop.loc[index_pop, 'Population_Total'] = aux_info_pop.loc[index_pop, str(int(i))]
aux_info_pop = aux_info_pop.loc[:,['Country_Code', 'Year', 'Population', 'Population_Total', 'Gdp']]
#fill
aux_info_pop['Population'] = aux_info_pop['Population'].apply(lambda x: x*1000)
index_nan = list(aux_info_pop[aux_info_pop['Population_Total'].isna()].index.values)
for i in index_nan:
    aux_info_pop.loc[i, 'Population_Total'] = aux_info_pop.loc[i, 'Population']

df_country_info = aux_info_pop.loc[:,['Country_Code', 'Year', 'Population_Total', 'Gdp']]
df_country_info.rename(columns={'Population_Total': 'Population'}, inplace=True)
df_country_info.fillna(np.nan, inplace=True)
#-----------------------------------------------------
#Energy without all types
df_energy_co2 = df_energy_co2[df_energy_co2['Energy_Type_Code'] > 0]
dim_energy = dim_energy[dim_energy['Energy_Type_Code'] > 0]
#-----------------------------------------------------

#Export csv
dim_country.to_csv("csv_export/dim_country.csv", index=False)
df_energy_co2.to_csv("csv_export/energyco2.csv", index=False)
dim_energy.to_csv("csv_export/dim_energy.csv", index=False)
df_energy_share.to_csv("csv_export/energy_share.csv", index=False)
df_energy_generation.to_csv("csv_export/energy_generation.csv", index=False)
df_energy_consum.to_csv("csv_export/energy_consumption.csv", index=False)
dim_fuel.to_csv("csv_export/dim_fuel.csv", index=False)
df_plant_generation.to_csv("csv_export/plant_generation.csv", index=False)
df_plant_info.to_csv("csv_export/plant_info.csv", index=False)
df_country_info.to_csv("csv_export/country_info.csv", index=False)
dim_calendar_year.to_csv("csv_export/dim_calendar_year.csv", index=False)