import pandas as pd
import string as s
from cmath import nan
import numpy as np

# Datasets
global_power = pd.read_csv('csv_default/global_power_plant_database.csv')
energyco2 = pd.read_csv('csv_default/energyco2.csv')
countryCode = pd.read_csv('csv_default/countries_codes_and_coordinates.csv')

# Standardization
global_power.loc[global_power.country_long ==
                 'Brunei Darussalam', 'country_long'] = 'Brunei'
global_power.loc[global_power.country_long ==
                 'Cote DIvoire', 'country_long'] = 'Ivory Coast'
global_power.loc[global_power.country_long ==
                 'Democratic Republic of the Congo', 'country_long'] = 'Congo'
global_power.loc[global_power.country_long ==
                 'Dominican Republic', 'country_long'] = 'Dominica'
global_power.loc[global_power.country_long == 'Iran',
                 'country_long'] = 'Iran Islamic Republic of'
global_power.loc[global_power.country_long == 'Laos',
                 'country_long'] = "Lao People's Democratic Republic"
global_power.loc[global_power.country_long == 'Macedonia',
                 'country_long'] = 'Macedonia the former Yugoslav Republic of'
global_power.loc[global_power.country_long == 'North Korea',
                 'country_long'] = "Korea Democratic People's Republic of"
global_power.loc[global_power.country_long == 'Palestine',
                 'country_long'] = 'Palestinian Territory Occupied'
global_power.loc[global_power.country_long == 'Tanzania',
                 'country_long'] = 'Tanzania United Republic of'
global_power.loc[global_power.country_long ==
                 'United States of America', 'country_long'] = 'United States'

# Auxiliar functions


def capword(list_values, sep='_'):
    for i in range(len(list_values)):
        list_values[i] = list_values[i].lower()
        list_values[i] = s.capwords(list_values[i], sep=sep)
    return list_values


def cap_namecolumns(df, sep='_'):
    col_old = list(df.columns.values)
    col_new = capword(list(df.columns.values), sep=sep)
    columns_name = dict(zip(col_old, col_new))
    df.rename(columns=columns_name, inplace=True)


# Country Code
code = countryCode['Alpha-3 code'].to_frame()
code.columns = ['code']
code['country'] = countryCode.Country.tolist()
code.drop_duplicates(subset=['code'], keep='last', inplace=True)
code.reset_index(inplace=True, drop=True)
cap_namecolumns(code)

# calendar
calendar = energyco2[['Year']].drop_duplicates()
calendar.reset_index(inplace=True, drop=True)
calendar.to_csv('calendar.csv')

# fuelType
fuelType = global_power[['country', 'country_long',
                         'primary_fuel']].drop_duplicates()
fuelType = code.join(fuelType.set_index('country_long'), on='Country')
fuelType = fuelType[['country', 'primary_fuel']]
fuelType.reset_index(inplace=True, drop=True)
cap_namecolumns(fuelType)
fuelType.columns = ['Code', 'Fuel']
fuelType.to_csv('fuelType.csv')

# plantInfo
plantInfo = global_power[['country_long', 'name',
                          'capacity_mw', 'latitude', 'longitude']].drop_duplicates()
plantInfo = code.join(plantInfo.set_index('country_long'), on='Country')
plantInfo = plantInfo.drop(['Country'], axis=1)
plantInfo.reset_index(inplace=True, drop=True)
plantInfo = plantInfo.fillna(nan)
cap_namecolumns(plantInfo)
plantInfo.to_csv('plantInfo.csv')

# plantGeneration
plantGeneration = global_power[['country', 'country_long', 'name', 'capacity_mw', 'latitude', 'longitude', 'primary_fuel', 'generation_gwh_2013', 'generation_gwh_2014', 'generation_gwh_2015', 'generation_gwh_2016', 'generation_gwh_2017', 'estimated_generation_gwh_2013', 'estimated_generation_gwh_2014',
                                'estimated_generation_gwh_2015', 'estimated_generation_gwh_2016', 'estimated_generation_gwh_2017', 'estimated_generation_note_2013', 'estimated_generation_note_2014', 'estimated_generation_note_2015', 'estimated_generation_note_2016',    'estimated_generation_note_2017']]
plantGeneration = code.join(
    plantGeneration.set_index('country_long'), on='Country')
plantGeneration = plantGeneration.drop(['Country', 'country'], axis=1)
cap_namecolumns(plantGeneration)
plantGeneration = plantGeneration.fillna(nan)
plantGeneration.to_csv('plantGeneration.csv')
