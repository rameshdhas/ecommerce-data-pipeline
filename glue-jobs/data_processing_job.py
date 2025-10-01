import sys
import boto3
import pandas as pd
import json
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import DataFrame
from typing import List, Dict, Any

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Get required job arguments
required_args = ['JOB_NAME', 'data_bucket']
optional_args = ['input_path', 'batch_mode']

# Get required arguments
args = getResolvedOptions(sys.argv, required_args)

# Try to get optional arguments
for opt_arg in optional_args:
    try:
        opt_args = getResolvedOptions(sys.argv, [opt_arg])
        args[opt_arg] = opt_args[opt_arg]
    except:
        args[opt_arg] = None

job.init(args['JOB_NAME'], args)

# Debug: Print the arguments received
print(f"Job initialized with arguments: {args}")
print(f"Data bucket: {args.get('data_bucket', 'NOT FOUND')}")

# Initialize AWS clients
# Bedrock Runtime is available in specific regions
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
# Use the default region for S3 operations
s3_client = boto3.client('s3')

def safe_float(value, default=0.0):
    """Safely convert a value to float, handling quoted strings and nulls"""
    if value is None or value == 'null' or value == '':
        return default
    try:
        # Remove quotes if present
        if isinstance(value, str):
            value = value.strip('"\'')
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int, handling quoted strings and nulls"""
    if value is None or value == 'null' or value == '':
        return default
    try:
        # Remove quotes if present
        if isinstance(value, str):
            value = value.strip('"\'')
        return int(float(value))  # Convert to float first to handle decimal strings
    except (ValueError, TypeError):
        return default

def generate_embeddings(text: str, model_id: str = "amazon.titan-embed-text-v1") -> List[float]:
    """
    Generate vector embeddings using Amazon Bedrock Titan model
    """
    try:
        # Titan embedding model v1 doesn't support dimensions parameter
        # It always returns 1536-dimensional embeddings
        body = json.dumps({
            "inputText": text
        })

        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())
        return response_body['embedding']

    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return []

def process_csv_data(input_path: str) -> DataFrame:
    """
    Read and process CSV data from S3
    """
    print(f"Processing CSV file: {input_path}")

    # Read CSV file using Glue DataCatalog or direct S3 read
    try:
        # Create dynamic frame from S3
        dynamic_frame = glueContext.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={
                "paths": [input_path],
                "recurse": True
            },
            format="csv",
            format_options={
                "withHeader": True,
                "separator": ","
            }
        )

        # Convert to Spark DataFrame
        df = dynamic_frame.toDF()
        print(f"Successfully read {df.count()} rows from CSV")
        return df

    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        raise

def create_text_content(row: Dict[str, Any]) -> str:
    """
    Create text content for embedding generation optimized for e-commerce semantic search
    """
    text_parts = []

    # Primary fields - most important for search
    if 'title' in row and row['title']:
        text_parts.append(f"Product: {row['title']}")

    if 'description' in row and row['description']:
        text_parts.append(f"Description: {row['description']}")

    if 'brand' in row and row['brand']:
        text_parts.append(f"Brand: {row['brand']}")

    if 'categories' in row and row['categories']:
        # Categories might be a list or string
        categories = row['categories']
        if isinstance(categories, str):
            text_parts.append(f"Categories: {categories}")
        else:
            text_parts.append(f"Categories: {', '.join(str(c) for c in categories)}")

    if 'features' in row and row['features']:
        features = row['features']
        if isinstance(features, str):
            text_parts.append(f"Features: {features}")
        else:
            text_parts.append(f"Features: {', '.join(str(f) for f in features)}")

    # Secondary fields - additional context
    if 'department' in row and row['department']:
        text_parts.append(f"Department: {row['department']}")

    if 'manufacturer' in row and row['manufacturer']:
        text_parts.append(f"Manufacturer: {row['manufacturer']}")

    if 'product_details' in row and row['product_details']:
        text_parts.append(f"Details: {row['product_details']}")

    if 'variations' in row and row['variations']:
        text_parts.append(f"Variations: {row['variations']}")

    # Context fields - enrich with signals
    if 'rating' in row and row['rating']:
        rating = safe_float(row['rating'])
        if rating >= 4.5:
            text_parts.append("Highly rated product")
        elif rating >= 4.0:
            text_parts.append("Well rated product")

    if 'reviews_count' in row and row['reviews_count']:
        reviews = safe_int(row['reviews_count'])
        if reviews > 1000:
            text_parts.append("Popular product with many reviews")
        elif reviews > 100:
            text_parts.append("Well-reviewed product")

    if 'availability' in row and row['availability']:
        if 'in stock' in str(row['availability']).lower():
            text_parts.append("Currently available")

    if 'discount' in row and row['discount']:
        discount_str = str(row['discount']).replace('%', '').strip('"\'')
        discount = safe_float(discount_str)
        if discount >= 50:
            text_parts.append("Significant discount available")
        elif discount >= 20:
            text_parts.append("On sale")

    # Price range information
    if 'final_price' in row and row['final_price']:
        text_parts.append(f"Price: {row['final_price']} {row.get('currency', 'USD')}")

    return " | ".join(text_parts)

def save_to_vector_store(processed_data: List[Dict[str, Any]]):
    """
    Save processed data with embeddings to vector database
    This is a placeholder - replace with your vector database implementation
    """
    try:
        # Convert to JSON for storage (replace with vector DB insert)
        output_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data": processed_data,
            "total_records": len(processed_data)
        }

        # Save to S3 as JSON (replace with vector database)
        bucket = args['data_bucket']
        key = f"processed-data/{pd.Timestamp.now().strftime('%Y/%m/%d')}/embeddings_{pd.Timestamp.now().strftime('%H%M%S')}.json"

        print(f"Attempting to save {len(processed_data)} records to s3://{bucket}/{key}")

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(output_data, indent=2),
            ContentType='application/json'
        )

        print(f"Successfully saved processed data to s3://{bucket}/{key}")

        # TODO: Replace this with actual vector database insertion
        # Examples:
        # - Amazon OpenSearch with vector search
        # - Pinecone
        # - Weaviate
        # - Chroma
        # - PostgreSQL with pgvector

    except Exception as e:
        print(f"Error saving to vector store: {str(e)}")
        print(f"Bucket: {bucket}, Key: {key}")
        raise

def main():
    """
    Main processing logic
    """
    # Check if input_path is provided
    input_path = args.get('input_path')

    if not input_path:
        # If no input path provided, process all CSV files in the data bucket
        bucket_name = args['data_bucket']
        print(f"No specific input path provided. Processing all CSV files in bucket: {bucket_name}")

        # List all CSV files in the bucket
        try:
            response = s3_client.list_objects_v2(Bucket=bucket_name)
            if 'Contents' in response:
                csv_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
                if csv_files:
                    # Process the first CSV file found (or you could process all)
                    input_path = f"s3://{bucket_name}/{csv_files[0]}"
                    print(f"Found {len(csv_files)} CSV files. Processing: {input_path}")
                else:
                    print("No CSV files found in the bucket")
                    return
            else:
                print("Bucket is empty")
                return
        except Exception as e:
            print(f"Error listing bucket contents: {str(e)}")
            return
    else:
        print(f"Starting data processing job for: {input_path}")

    # Read and process CSV data
    df = process_csv_data(input_path)

    # Convert to Pandas for easier processing
    pandas_df = df.toPandas()

    processed_records = []

    # Track statistics
    successful_embeddings = 0
    failed_embeddings = 0

    # Process each row
    for index, row in pandas_df.iterrows():
        try:
            # Create text content for embedding
            text_content = create_text_content(row.to_dict())

            if text_content:
                # Generate embeddings
                embeddings = generate_embeddings(text_content)

                # For now, continue even if embeddings fail (store record without embeddings)
                # This allows us to at least save the processed metadata
                if embeddings or True:  # Always process records even without embeddings
                    # Extract metadata for filtering in vector search
                    metadata = {
                        "asin": row.get('asin', ''),
                        "brand": row.get('brand', ''),
                        "seller_name": row.get('seller_name', ''),
                        "categories": row.get('categories', ''),
                        "department": row.get('department', ''),
                        "rating": safe_float(row.get('rating', 0)),
                        "reviews_count": safe_int(row.get('reviews_count', 0)),
                        "final_price": safe_float(row.get('final_price', 0)),
                        "currency": row.get('currency', 'USD'),
                        "availability": row.get('availability', ''),
                        "discount": row.get('discount', ''),
                        "is_available": row.get('is_available', False),
                        "bought_past_month": safe_int(row.get('bought_past_month', 0)),
                        "timestamp": row.get('timestamp', ''),
                    }

                    # Track embedding success/failure
                    if embeddings:
                        successful_embeddings += 1
                    else:
                        failed_embeddings += 1

                    # Create processed record
                    processed_record = {
                        "id": row.get('asin', f"record_{index}"),  # Use ASIN as ID if available
                        "title": row.get('title', ''),
                        "url": row.get('url', ''),
                        "image_url": row.get('image_url', ''),
                        "text_content": text_content,
                        "embeddings": embeddings if embeddings else [],
                        "embedding_dimension": len(embeddings) if embeddings else 0,
                        "has_embeddings": bool(embeddings),
                        "metadata": metadata,
                        "original_data": row.to_dict()
                    }
                    processed_records.append(processed_record)

                    if (index + 1) % 10 == 0:
                        print(f"Processed {index + 1} records (Embeddings: {successful_embeddings} success, {failed_embeddings} failed)...")

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            continue

    print(f"Successfully processed {len(processed_records)} records")
    print(f"Embeddings generated: {successful_embeddings} successful, {failed_embeddings} failed")

    # Save to vector store
    if processed_records:
        save_to_vector_store(processed_records)
        print(f"Data saved to S3. Records with embeddings: {successful_embeddings}/{len(processed_records)}")
    else:
        print("WARNING: No records were processed!")

    print("Data processing job completed successfully!")

if __name__ == "__main__":
    main()
    job.commit()