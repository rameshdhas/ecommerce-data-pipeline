import json
import boto3
import os

s3_client = boto3.client('s3')
glue_client = boto3.client('glue')

def handler(event, context):
    bucket_name = os.environ['DATA_BUCKET']
    glue_job_name = os.environ['GLUE_JOB_NAME']

    try:
        # Process all CSV files in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        csv_count = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.csv'):
                        # Start a single Glue job to process all files
                        # Or start individual jobs for each file
                        csv_count += 1

        if csv_count > 0:
            # Start a single Glue job to process all CSV files
            response = glue_client.start_job_run(
                JobName=glue_job_name,
                Arguments={
                    '--input-path': f's3://{bucket_name}/',
                    '--batch-mode': 'true'
                }
            )
            print(f"Started batch Glue job {response['JobRunId']} for {csv_count} CSV files")

        return {
            'statusCode': 200,
            'body': json.dumps(f'Processed {csv_count} CSV files')
        }

    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        raise