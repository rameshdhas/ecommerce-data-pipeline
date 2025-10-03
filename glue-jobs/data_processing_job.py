import sys
from datetime import datetime
from awsglue.transforms import *
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *
from typing import List, Dict, Any
import json

# Import our modular components
from shared.config import get_job_arguments
from shared.aws_clients import get_aws_clients
from shared.utils import safe_float, safe_int
from embedding.generator import generate_embeddings_batch
from processing.csv_reader import process_csv_data
from processing.text_processor import create_text_content, extract_clean_description
from storage.s3_writer import save_to_s3_batch
from storage.elasticsearch_writer import save_to_elasticsearch_batch

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

# Configuration for scalable processing
BATCH_SIZE = int(args.get('batch_size', 1000))  # Process in batches
PARTITION_SIZE = int(args.get('partition_size', 10000))  # Spark partition size
MAX_EMBEDDING_BATCH = int(args.get('max_embedding_batch', 25))  # Bedrock batch limit


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


def create_processing_udf(embedding_model: str, batch_size: int):
    """Create a UDF for processing batches of records"""

    def process_batch_records(iterator):
        """Process records in batches to optimize API calls"""
        from embedding.generator import generate_embeddings_batch
        from processing.text_processor import create_text_content, extract_clean_description
        import json

        batch = []
        results = []

        for row in iterator:
            batch.append(row)

            # Process when batch is full or at end of iterator
            if len(batch) >= batch_size:
                processed_batch = process_record_batch(batch, embedding_model)
                results.extend(processed_batch)
                batch = []

        # Process remaining records
        if batch:
            processed_batch = process_record_batch(batch, embedding_model)
            results.extend(processed_batch)

        return iter(results)

    return process_batch_records


def process_record_batch(batch: List[Any], embedding_model: str) -> List[Dict]:
    """Process a batch of records efficiently"""
    processed_records = []

    # Extract text content for all records
    text_contents = []
    record_data = []

    for row in batch:
        row_dict = row.asDict()
        text_content = create_text_content(row_dict)
        clean_description = extract_clean_description(row_dict)

        text_contents.append(text_content)
        record_data.append({
            'row_dict': row_dict,
            'text_content': text_content,
            'clean_description': clean_description
        })

    # Generate embeddings in batch (more efficient API usage)
    try:
        embeddings_batch = generate_embeddings_batch(text_contents, model_id=embedding_model)
    except Exception as e:
        print(f"Batch embedding generation failed: {e}")
        embeddings_batch = [[] for _ in text_contents]  # Empty embeddings for all

    # Create processed records
    for i, record_info in enumerate(record_data):
        row_dict = record_info['row_dict']
        embeddings = embeddings_batch[i] if i < len(embeddings_batch) else []

        # Convert timestamp to ISO format
        timestamp_value = row_dict.get('timestamp', '')
        if timestamp_value:
            try:
                timestamp_iso = datetime.fromisoformat(str(timestamp_value)).isoformat()
            except:
                timestamp_iso = datetime.now().isoformat()
        else:
            timestamp_iso = datetime.now().isoformat()

        # Create metadata
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

        processed_record = {
            "id": row_dict.get('asin', f"record_{i}"),
            "title": row_dict.get('title', ''),
            "description": record_info['clean_description'],
            "url": row_dict.get('url', ''),
            "image_url": row_dict.get('image_url', ''),
            "text_content": record_info['text_content'],
            "embeddings": embeddings,
            "embedding_dimension": len(embeddings) if embeddings else 0,
            "has_embeddings": bool(embeddings),
            "metadata": metadata,
            "original_data": row_dict
        }

        processed_records.append(processed_record)

    return processed_records


def save_partition_data(iterator, data_bucket: str, elasticsearch_endpoint: str, elasticsearch_api_key: str, embedding_model: str):
    """Save data from a partition to S3 and Elasticsearch"""
    records = list(iterator)
    if not records:
        return iter([])

    # Save to S3 in batch
    s3_success = save_to_s3_batch(records, data_bucket)

    # Save to Elasticsearch in batch
    es_success = save_to_elasticsearch_batch(
        records,
        elasticsearch_endpoint,
        elasticsearch_api_key,
        embedding_model
    )

    # Return processing results
    results = []
    for record in records:
        results.append({
            'record_id': record.get('id', ''),
            's3_success': s3_success,
            'es_success': es_success,
            'has_embeddings': record.get('has_embeddings', False)
        })

    return iter(results)


def main():
    """Scalable main processing logic"""
    input_path = args.get('input_path')

    if not input_path:
        bucket_name = args['data_bucket']
        print(f"Processing all CSV files in bucket: {bucket_name}")
        input_path = find_input_file(bucket_name)
    else:
        print(f"Starting scalable data processing for: {input_path}")

    embedding_model = args.get('embedding_model', 'amazon.titan-embed-text-v1')
    print(f"Using embedding model: {embedding_model} with batch size: {BATCH_SIZE}")

    # Read CSV data using Glue's DynamicFrame for better performance
    datasource = glueContext.create_dynamic_frame.from_options(
        format_options={"multiline": False},
        connection_type="s3",
        format="csv",
        connection_options={
            "paths": [input_path],
            "recurse": True
        },
        transformation_ctx="datasource"
    )

    # Convert to DataFrame for Spark operations
    df = datasource.toDF()

    # Repartition for optimal processing (avoid too many small partitions)
    total_rows = df.count()
    optimal_partitions = max(1, min(total_rows // PARTITION_SIZE, 200))  # Cap at 200 partitions
    df = df.repartition(optimal_partitions)

    print(f"Processing {total_rows} records across {optimal_partitions} partitions")

    # Process data using Spark's mapPartitions for scalability
    processed_rdd = df.rdd.mapPartitions(
        lambda iterator: create_processing_udf(embedding_model, BATCH_SIZE)(iterator)
    )

    # Save processed data using mapPartitions to handle large datasets
    save_results_rdd = processed_rdd.mapPartitions(
        lambda iterator: save_partition_data(
            iterator,
            args['data_bucket'],
            args.get('elasticsearch_endpoint', ''),
            args.get('elasticsearch_api_key', ''),
            embedding_model
        )
    )

    # Collect results to trigger execution
    results = save_results_rdd.collect()

    # Calculate statistics
    total_processed = len(results)
    successful_embeddings = sum(1 for r in results if r.get('has_embeddings', False))
    s3_successes = sum(1 for r in results if r.get('s3_success', False))
    es_successes = sum(1 for r in results if r.get('es_success', False))

    print(f"Processing complete:")
    print(f"  Total records processed: {total_processed}")
    print(f"  Records with embeddings: {successful_embeddings}")
    print(f"  S3 saves successful: {s3_successes}")
    print(f"  Elasticsearch saves successful: {es_successes}")

    # Log processing efficiency
    if total_processed > 0:
        embedding_rate = (successful_embeddings / total_processed) * 100
        print(f"  Embedding success rate: {embedding_rate:.1f}%")


if __name__ == "__main__":
    main()
    job.commit()