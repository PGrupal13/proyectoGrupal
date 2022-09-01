# Api ejecuta query sql y devuelve un dataframe
import sys
import pandas as pd
import numpy as np
import requests as req
import io

url = '''https://3wecq7h5ea.execute-api.us-east-2.amazonaws.com/prod/nn?query=


SELECT *
FROM main_db.country_info

left join main_db.dim_country 
on main_db.dim_country.country_code = main_db.country_info.country_code

where main_db.country_info.country_code   = ' ARG'
limit 50;
 
'''

   
response = req.get (url )
if response.status_code == 200:
    data = response
    
else: 
    print(response.status_code)
    sys.exit()
    
    
df = pd.read_csv(io.BytesIO(data.content))
