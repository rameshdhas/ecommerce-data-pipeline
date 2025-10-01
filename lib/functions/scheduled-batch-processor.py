import json
import boto3
import os
from datetime import datetime, timedelta

s3_client = boto3.client('s3')
glue_client = boto3.client('glue')

def handler(event, context):
    bucket_name = os.environ['DATA_BUCKET']
    glue_job_name = os.environ['GLUE_JOB_NAME']

    # Get yesterday's date for processing
    yesterday = datetime.now() - timedelta(days=1)
    prefix = f"daily-uploads/{yesterday.strftime('%Y/%m/%d')}/"

    print(f"Processing files from prefix: {prefix}")

    try:
        # List all CSV files in the daily upload folder
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )

        if 'Contents' not in response:
            print(f"No files found for {yesterday.strftime('%Y-%m-%d')}")
            return {
                'statusCode': 200,
                'body': json.dumps('No files to process')
            }

        csv_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]

        for csv_file in csv_files:
            print(f"Starting Glue job for: {csv_file}")

            # Start Glue job for each CSV file
            glue_response = glue_client.start_job_run(
                JobName=glue_job_name,
                Arguments={
                    '--input-path': f's3://{bucket_name}/{csv_file}',
                    '--data_bucket': bucket_name
                }
            )
            print(f"Started Glue job {glue_response['JobRunId']} for {csv_file}")

        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully triggered {len(csv_files)} Glue jobs')
        }

    except Exception as e:
        print(f"Error in scheduled batch processing: {str(e)}")
        raise