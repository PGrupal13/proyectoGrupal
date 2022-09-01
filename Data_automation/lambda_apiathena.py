import boto3
import time
import logging
import json
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)
client_1 = boto3.client('s3')

def execute_athena_query(parameters, wait=True):
    """
    function to execure the query against in AWS athena table
    :param parameters: 
    :param wait: 
    :return: 
    """
    try:
        error_message = ''
        location = ''
        row_values = []
        logger.info("function execution starts")
        session = boto3.Session()
        client = session.client('athena')

        # This function executes the query and returns the query execution ID
        response_query_execution_id = client.start_query_execution(
            QueryString=parameters['query'],
            QueryExecutionContext={
                'Database': parameters['database']
            },
            ResultConfiguration={
                'OutputLocation': 's3://' + parameters['bucket'] + '/' + parameters['path']
            }
        )

        if not wait:
            return response_query_execution_id['QueryExecutionId']
        else:
            response_get_query_details = client.get_query_execution(
                QueryExecutionId=response_query_execution_id['QueryExecutionId']
            )
            status = 'RUNNING'
            iterations = 360  # 30 mins

            while iterations > 0:
                iterations = iterations - 1
                response_get_query_details = client.get_query_execution(
                    QueryExecutionId=response_query_execution_id['QueryExecutionId']
                )
                status = response_get_query_details['QueryExecution']['Status']['State']

                if (status == 'FAILED') or (status == 'CANCELLED'):
                    error_reason = response_get_query_details['QueryExecution']['Status']['StateChangeReason']
                    logger.error("Function Name: {} Error: {}"
                                 .format(execute_athena_query.__name__, str(error_reason)))
                    error_message = error_reason

                elif status == 'SUCCEEDED':
                    location = response_get_query_details['QueryExecution']['ResultConfiguration']['OutputLocation']
                    

                    response_query_result = client.get_query_results(
                        QueryExecutionId=response_query_execution_id['QueryExecutionId']
                    )
                    rows = response_query_result['ResultSet']['Rows'][1:]
                    row_values = [obj['Data'][0]['VarCharValue'] for obj in rows]
                    break
                else:
                    time.sleep(5)
    except Exception as exception:
        logger.error("Function Name: {} Error: {}"
                     .format(execute_athena_query.__name__, str(exception)))
        error_message = exception
        
    return location, row_values, error_message


def lambda_handler(event, context):
   
    queryparam1= event["queryStringParameters"]['query']
    sql_query_to_execute = queryparam1
    athena_parameters = {
            'region': 'us-east-2',
            'database': 'default',
            'bucket': 'query-python-output',
            'path': 'athena_demo/output/',
            'query': sql_query_to_execute
        }
    location, result_set, error_message = execute_athena_query(athena_parameters)
    print(f' location = {location}')
    filename = re.findall('.*\/(.*)', location)[0]
    
    resp = client_1.get_object(
    Bucket=athena_parameters['bucket'],
    Key=athena_parameters['path']+filename,)
    
    
    print(f' error mesasge = {error_message}')
    data= resp['Body'].read()
   
    
    return {
        'statusCode': 200,
        'body': data
    }