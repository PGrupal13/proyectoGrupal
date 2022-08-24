import numpy as np
import pandas as pd
import string as s
from sklearn import preprocessing

#Datasets
df_energy_consumption = pd.read_csv('csv_default/owid-energy-consumption-source.csv')
df_energyco2 = pd.read_csv('csv_default/energyco2.csv')
dim_country = pd.read_csv('csv_default/dim_country.csv')

#Auxiliar functions
def capword(list_values, sep = '_'):
    for i in range(len(list_values)):
        list_values[i] = list_values[i].lower()
        list_values[i] = s.capwords(list_values[i], sep=sep)
    return list_values

def cap_namecolumns(df, sep = '_'):
    col_old = list(df.columns.values)
    col_new = capword(list(df.columns.values), sep=sep)
    columns_name = dict(zip(col_old, col_new))
    df.rename(columns = columns_name, inplace=True)

def rename_country(df, list_country_old, list_country_new):
    for i in range(len(list_country_old)):
        index = df[df['Country'] == list_country_old[i]].index.values
        df.loc[index, 'Country'] = list_country_new[i]

#Rename columns
cap_namecolumns(df_energy_consumption)
cap_namecolumns(df_energyco2)

#Rename countries
#df_energyco2
l1 = ['The Bahamas', 'British Virgin Islands', 'Cabo Verde', 'Congo-Brazzaville',
      'Congo-Kinshasa', 'Côte d’Ivoire', 'Falkland Islands', 'Gambia,T he', 
      'Iran', 'North Korea', 'Laos', 
      'Macau', 'Moldova', 'Micronesia', 'Palestinian Territories', 
      'Reunion', 'Saint Helena', 'Saint Vincent/Grenadines', 
      'Eswatini', 'Syria', 'Tanzania', 'U.S. Virgin Islands', 
      'North Macedonia']

l2 = ['Bahamas', 'Virgin Islands, British', 'Cape Verde', 'Congo', 
      'Congo, the Democratic Republic of the', "Cote d'Ivoire", 'Falkland Islands (Malvinas)', 'Gambia', 
      'Iran, Islamic Republic of', "Korea, Democratic People's Republic of", "Lao People's Democratic Republic", 
      'Macao', 'Moldova, Republic of', 'Micronesia, Federated States of', 'Palestinian Territory, Occupied', 
      'Réunion', 'Saint Helena,  Ascension and Tristan da Cunha', 'Saint Vincent and the Grenadines', 
      'Swaziland', 'Syrian Arab Republic', 'Tanzania, United Republic of', 'Virgin Islands, U.S.', 
      'Macedonia, Republic of North']

rename_country(df_energyco2, l1, l2)

#df_energy_consumption
l3 = ['Czechia', 'Democratic Republic of Congo', 'Falkland Islands', 
      'Faeroe Islands', 'Iran', 'North Korea', 
      'Laos', 'North Macedonia', 'Micronesia (country)', 
      'Moldova', 'Palestine', 'Reunion', 'Eswatini',
      'Syria', 'Tanzania', 'Timor', 'United States Virgin Islands',
      'British Virgin Islands', 'Saint Helena']
    
l4 = ['Czech Republic', 'Congo, the Democratic Republic of the', 'Falkland Islands (Malvinas)',
      'Faroe Islands', 'Iran, Islamic Republic of',  "Korea, Democratic People's Republic of",
      "Lao People's Democratic Republic", 'Macedonia, Republic of North', 'Micronesia, Federated States of', 
      'Moldova,  Republic of', 'Palestinian Territory, Occupied', 'Réunion', 'Swaziland', 
      'Syrian Arab Republic', 'Tanzania, United Republic of', 'Timor-Leste', 'Virgin Islands, U.S.',
      'Virgin Islands, British', 'Saint Helena, Ascension and Tristan da Cunha']

rename_country(df_energy_consumption, l3, l4)

#Country filter
df_energy_consumption = df_energy_consumption[df_energy_consumption['Country'].isin(dim_country['Country'].values)]
df_energyco2 = df_energyco2[df_energyco2['Country'].isin(dim_country['Country'].values)]


#Energy(dim)
le = preprocessing.LabelEncoder()
df_energyco2['Energy_Type_Code'] = le.fit_transform(df_energyco2['Energy_Type'])
dim_energy = pd.DataFrame(le.classes_)
dim_energy.reset_index(inplace=True)
dim_energy.rename(columns = {'index': 'Code', 0: 'Energy_Type'}, inplace=True)
dim_energy['Energy_Type'] = dim_energy['Energy_Type'].apply(lambda x: x.replace('_', ' ').capitalize())

#Merge country code
df_energy_consumption = pd.merge(dim_country, df_energy_consumption, on='Country', how='left')
df_energyco2 = pd.merge(dim_country, df_energyco2, on='Country', how='left')

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
df_energy_consumption = df_energy_consumption.loc[:, columns_energy_consumption]

#Country info
df_country_info = df_energyco2.loc[:, ['Country_Code', 'Year', 'Gdp', 'Population']]

#Null values
df_energyco2.fillna(np.nan, inplace=True)
df_energy_share.fillna(np.nan, inplace=True)
df_energy_generation.fillna(np.nan, inplace=True)
df_energy_consumption.fillna(np.nan, inplace=True)
df_country_info.fillna(np.nan, inplace=True)

#Export csv
# df_energy_co2.to_csv("energyco2.csv", index=False)
# dim_energy.to_csv("dim_energy.csv", index=False)
# df_energy_share.to_csv("energy_share.csv", index=False)
# df_energy_generation.to_csv("energy_generation.csv", index=False)
# df_energy_consumption.to_csv("energy_consumption.csv", index=False)
# df_country_info.to_csv("country_info.csv", index=False)