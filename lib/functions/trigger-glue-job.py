import json
import boto3
import os

glue_client = boto3.client('glue')

def handler(event, context):
    glue_job_name = os.environ['GLUE_JOB_NAME']

    # Parse S3 event
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        # Only process CSV files
        if key.endswith('.csv'):
            try:
                print(f"Processing file: {key}")

                # Start Glue job with scalable configuration
                response = glue_client.start_job_run(
                    JobName=glue_job_name,
                    Arguments={
                        '--input_path': f's3://{bucket}/{key}',
                        '--data_bucket': bucket
                    }
                )

                print(f"Started Glue job {response['JobRunId']} for {key}")

            except Exception as e:
                print(f"Error starting Glue job: {str(e)}")
                raise

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully triggered Glue job')
    }