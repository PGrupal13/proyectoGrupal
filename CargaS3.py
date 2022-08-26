from csv import excel_tab
from distutils.command.upload import upload
from dotenv import load_dotenv
import os
import logging
import boto3
from botocore.exceptions import ClientError
import os

#Importa las variables de entorno
load_dotenv()

def upload_file(file_name, bucket, object_name=None):
    """Función que realiza la carga de archivos a S3

    Args:
        file_name (String): Archivo a cargar
        bucket (String): Bucket al que se va a cargar
        object_name (String, optional): Nombre que se le va a dar al archivo en S3, si no se especifica, entonces file_name será usado
    return: True si el archivo fue cargado, en caso contrario retorna False
    """

    #Credenciales
    access_id = os.environ.get('ACCESS_ID')
    secret_access_key = os.environ.get('SECRET_ACCESS_KEY')

    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3', aws_access_key_id=access_id, aws_secret_access_key=secret_access_key)
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

for archivo in os.listdir('D:\\Documentos\\Henry\\ProyectoFinal\\Proyecto\\proyectoGrupal\\csv_export'):
    if '.csv' in archivo:
        print(f'Cargando archivo: {archivo}')
        archivo_dir = './csv_export/' + archivo
        file_key = 'data/' + str(archivo) + '/' + str(archivo)
        upload_file(archivo_dir, 'databasealejandro', file_key)
