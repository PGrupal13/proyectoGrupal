import re
import boto3
import pandas
import time
import csv
from dotenv import load_dotenv
import os
import io
import pandas as pd

load_dotenv()

access_id = os.environ('ACCESS_ID')
secret_access_key = os.environ('SECRET_ACCESS_KEY')

params = {
    'region': 'us-east-2',
    'database': '', #base de datos de athena
    'bucket': '', #Bucket de S3
    'path': 'script', #Carpeta de salida en S3
    'query': '' #query a ejecutar
}

session = boto3.Session()

def athena_query(client, params):
    """Función que envía la query a athena con los detalles y recive un objeto de ejecución
    """
    response = client.start_query_execution(
        QueryString=params["query"],
        QueryExecutionContext={
            'Database': params['database']
        },
        ResultConfiguration={
            'OutputLocation': 's3://' + params['bucket'] + '/' + params['path']
        }
    )
    return response

def athena_to_s3(session, params, max_execution = 5):
    """Función que realiza los siguientes pasos:
    1. Envía la query a Athena
    2. Realiza un sondeo de los resultados obtenidos
    3. Retorna el archivo de S3 en el que se guarda el resultado de la query
    4. Descarga el archivo a una variable temporal y retorna un dataframe
    """
    client = session.client('athena',
                            aws_access_key_id = params['access_key'],
                            aws_secret_access_key = params['secret_key'],
                            region_name=params["region"])
    execution = athena_query(client, params)
    execution_id = execution['QueryExecutionId']
    state = 'RUNNING'

    while (max_execution > 0 and state in ['RUNNING', 'QUEUED']):
        max_execution = max_execution - 1
        response = client.get_query_execution(QueryExecutionId = execution_id)

        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']
            if state == 'FAILED':
                return False
            elif state == 'SUCCEEDED':
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*\/(.*)', s3_path)[0]  

                # Descarga del CSV
                temp_file_location: str = "athena_query_result.csv"
                s3_client = boto3.client(
                  "s3",
                  aws_access_key_id = params['access_key'],
                  aws_secret_access_key = params['secret_key'],
                  region_name=params["region"]
                )
                s3_client.download_file(params['bucket'],
                                        f"{params['path']}/{filename}",
                                        temp_file_location
                )
                return pd.read_csv(temp_file_location)

        time.sleep(1)
   
    return False