import sys
import pandas as pd
from datetime import datetime
from awsglue.transforms import *
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from typing import List, Dict, Any

# Import our modular components
from shared.config import get_job_arguments
from shared.aws_clients import get_aws_clients
from shared.utils import safe_float, safe_int
from embedding.generator import generate_embeddings
from processing.csv_reader import process_csv_data
from processing.text_processor import create_text_content, extract_clean_description
from storage.s3_writer import save_to_s3
from storage.elasticsearch_writer import save_to_elasticsearch

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Get job arguments
args = get_job_arguments()
job.init(args['JOB_NAME'], args)

# Initialize AWS clients
aws_clients = get_aws_clients()

# Simplified processing - text-only embeddings
print("Image analysis disabled - using text-only processing")


def find_input_file(bucket_name: str) -> str:
    """Find the first CSV file in the bucket if no input path is specified"""
    try:
        response = aws_clients.s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            csv_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
            if csv_files:
                input_path = f"s3://{bucket_name}/{csv_files[0]}"
                print(f"Found {len(csv_files)} CSV files. Processing: {input_path}")
                return input_path
            else:
                raise ValueError("No CSV files found in the bucket")
        else:
            raise ValueError("Bucket is empty")
    except Exception as e:
        print(f"Error listing bucket contents: {str(e)}")
        raise


def process_record(row_dict: Dict[str, Any], index: int, embedding_model: str) -> Dict[str, Any]:
    """Process a single record and generate embeddings"""
    # Create text content for embedding
    text_content = create_text_content(row_dict)

    # Extract clean description
    clean_description = extract_clean_description(row_dict)

    # Generate embeddings if text content exists
    embeddings = []
    if text_content:
        embeddings = generate_embeddings(text_content, model_id=embedding_model)

    # Convert timestamp to ISO format for Elasticsearch
    timestamp_value = row_dict.get('timestamp', '')
    if timestamp_value:
        try:
            if isinstance(timestamp_value, str):
                dt = pd.to_datetime(timestamp_value)
                timestamp_iso = dt.isoformat()
            else:
                timestamp_iso = pd.to_datetime(timestamp_value).isoformat()
        except Exception as e:
            print(f"Warning: Could not parse timestamp '{timestamp_value}': {e}")
            timestamp_iso = datetime.now().isoformat()
    else:
        timestamp_iso = datetime.now().isoformat()

    # Create metadata for filtering in vector search
    metadata = {
        "asin": row_dict.get('asin', ''),
        "brand": row_dict.get('brand', ''),
        "seller_name": row_dict.get('seller_name', ''),
        "categories": row_dict.get('categories', ''),
        "department": row_dict.get('department', ''),
        "rating": safe_float(row_dict.get('rating', 0)),
        "reviews_count": safe_int(row_dict.get('reviews_count', 0)),
        "final_price": safe_float(row_dict.get('final_price', 0)),
        "currency": row_dict.get('currency', 'USD'),
        "availability": row_dict.get('availability', ''),
        "discount": row_dict.get('discount', ''),
        "is_available": row_dict.get('is_available', False),
        "bought_past_month": safe_int(row_dict.get('bought_past_month', 0)),
        "timestamp": timestamp_iso,
    }

    # Create processed record
    processed_record = {
        "id": row_dict.get('asin', f"record_{index}"),
        "title": row_dict.get('title', ''),
        "description": clean_description,
        "url": row_dict.get('url', ''),
        "image_url": row_dict.get('image_url', ''),
        "text_content": text_content,
        "embeddings": embeddings if embeddings else [],
        "embedding_dimension": len(embeddings) if embeddings else 0,
        "has_embeddings": bool(embeddings),
        "metadata": metadata,
        "original_data": row_dict
    }

    return processed_record


def main():
    """Main processing logic"""
    # Check if input_path is provided
    input_path = args.get('input_path')

    if not input_path:
        # If no input path provided, process all CSV files in the data bucket
        bucket_name = args['data_bucket']
        print(f"No specific input path provided. Processing all CSV files in bucket: {bucket_name}")
        input_path = find_input_file(bucket_name)
    else:
        print(f"Starting data processing job for: {input_path}")

    # Read and process CSV data
    df = process_csv_data(input_path, glueContext)

    # Convert to Pandas for easier processing
    pandas_df = df.toPandas()

    processed_records = []

    # Track statistics
    successful_embeddings = 0
    failed_embeddings = 0

    # Get the embedding model to use (default to Titan v1 if not specified)
    embedding_model = args.get('embedding_model', 'amazon.titan-embed-text-v1')
    print(f"Using embedding model: {embedding_model}")

    # Process each row
    for index, row in pandas_df.iterrows():
        try:
            processed_record = process_record(row.to_dict(), index, embedding_model)

            # Track embedding success/failure
            if processed_record['has_embeddings']:
                successful_embeddings += 1
            else:
                failed_embeddings += 1

            processed_records.append(processed_record)

            if (index + 1) % 10 == 0:
                print(f"Processed {index + 1} records (Embeddings: {successful_embeddings} success, {failed_embeddings} failed)...")

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            continue

    print(f"Successfully processed {len(processed_records)} records")
    print(f"Embeddings generated: {successful_embeddings} successful, {failed_embeddings} failed")

    # Save to S3 and Elasticsearch
    if processed_records:
        # Save to S3
        s3_success = save_to_s3(processed_records, args['data_bucket'])

        # Save to Elasticsearch
        es_success = save_to_elasticsearch(
            processed_records,
            args.get('elasticsearch_endpoint', ''),
            args.get('elasticsearch_api_key', ''),
            embedding_model
        )

        if s3_success and es_success:
            print("Data successfully saved to both S3 and Elasticsearch")
        elif s3_success:
            print("Data saved to S3, but Elasticsearch indexing failed or was skipped")
        else:
            print("Warning: Failed to save data to storage systems")

        print(f"Processing complete. Records with embeddings: {successful_embeddings}/{len(processed_records)}")
    else:
        print("WARNING: No records were processed!")

    print("Data processing job completed successfully!")


if __name__ == "__main__":
    main()
    job.commit()