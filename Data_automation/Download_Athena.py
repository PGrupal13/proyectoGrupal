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

access_id = os.environ.get('ACCESS_ID')
secret_access_key = os.environ.get('SECRET_ACCESS_KEY')

params = {
    'region': 'us-east-2',
    'database': 'au-eze', #base de datos de athena
    'bucket': 'databasealejandro', #Bucket de S3
    'path': 'script', #Carpeta de salida en S3
    'query': 'SELECT * FROM country_info' #query a ejecutar
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
    """
    client = session.client('athena', region_name=params["region"], aws_access_key_id = access_id, aws_secret_access_key=secret_access_key)
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
                raise Exception('Fallo la consulta')
                return False
            elif state == 'SUCCEEDED':
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*\/(.*)', s3_path)[0]
                return filename
        time.sleep(1)
    
    return False

def s3_to_pandas(session, params, s3_filename):
    """
    Función que crea un dataframe a partir de los datos encontrados en S3

    Args:
        session (aws object): Sesión del usuario
        params (dic): Parametros del query
        s3_filename (string): Bucket de S3

    Returns:
        DataFrame: Dataframe de pandas
    """
    s3client = session.client('s3',aws_access_key_id = access_id, aws_secret_access_key=secret_access_key)
    obj = s3client.get_object(Bucket=params['bucket'],
                              Key=params['path'] + '/' + s3_filename)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df


def traer_df(myquery):
    # Query Athena and get the s3 filename as a result
    """
    Función que envía una query a Athena y retorna un dataframe
    con la información encontrada

    Args:
        myquery (Sentencia SQL): Sentencia SQL

    Returns:
        DataFrame: DataFrame 
    """
    params.update({'query': myquery})
    s3_filename = athena_to_s3(session, params)
    mydf=s3_to_pandas(session, params, s3_filename)
    cleanup(session, params)
    return mydf


def cleanup(session, params):
    """
    Función que borra los archivos del bucket
    """
    s3 = session.resource('s3',aws_access_key_id = access_id, aws_secret_access_key=secret_access_key)
    my_bucket = s3.Bucket(params['bucket'])
    for item in my_bucket.objects.filter(Prefix=params['path']):
        item.delete()


country_info = traer_df('SELECT * FROM dim_country')
print(country_info.head())