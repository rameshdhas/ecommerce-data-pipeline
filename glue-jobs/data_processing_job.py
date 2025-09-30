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

# Get job arguments
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'input-path',
    'data-bucket'
])

job.init(args['JOB_NAME'], args)

# Initialize AWS clients
bedrock_client = boto3.client('bedrock-runtime')
s3_client = boto3.client('s3')

def generate_embeddings(text: str, model_id: str = "amazon.titan-embed-text-v1") -> List[float]:
    """
    Generate vector embeddings using Amazon Bedrock Titan model
    """
    try:
        body = json.dumps({
            "inputText": text,
            "dimensions": 1536,  # Titan embeddings dimension
            "normalize": True
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
        try:
            rating = float(row['rating'])
            if rating >= 4.5:
                text_parts.append("Highly rated product")
            elif rating >= 4.0:
                text_parts.append("Well rated product")
        except:
            pass

    if 'reviews_count' in row and row['reviews_count']:
        try:
            reviews = int(row['reviews_count'])
            if reviews > 1000:
                text_parts.append("Popular product with many reviews")
            elif reviews > 100:
                text_parts.append("Well-reviewed product")
        except:
            pass

    if 'availability' in row and row['availability']:
        if 'in stock' in str(row['availability']).lower():
            text_parts.append("Currently available")

    if 'discount' in row and row['discount']:
        try:
            discount = float(str(row['discount']).replace('%', ''))
            if discount >= 50:
                text_parts.append("Significant discount available")
            elif discount >= 20:
                text_parts.append("On sale")
        except:
            pass

    # Price range information
    if 'final_price' in row and row['final_price']:
        text_parts.append(f"Price: {row['final_price']} {row.get('currency', 'USD')}")

    return " | ".join(text_parts)

def save_to_vector_store(processed_data: List[Dict[str, Any]], output_path: str):
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
        bucket = args['data-bucket']
        key = f"processed-data/{pd.Timestamp.now().strftime('%Y/%m/%d')}/embeddings_{pd.Timestamp.now().strftime('%H%M%S')}.json"

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(output_data, indent=2),
            ContentType='application/json'
        )

        print(f"Saved processed data to s3://{bucket}/{key}")

        # TODO: Replace this with actual vector database insertion
        # Examples:
        # - Amazon OpenSearch with vector search
        # - Pinecone
        # - Weaviate
        # - Chroma
        # - PostgreSQL with pgvector

    except Exception as e:
        print(f"Error saving to vector store: {str(e)}")
        raise

def main():
    """
    Main processing logic
    """
    input_path = args['input-path']
    print(f"Starting data processing job for: {input_path}")

    # Read and process CSV data
    df = process_csv_data(input_path)

    # Convert to Pandas for easier processing
    pandas_df = df.toPandas()

    processed_records = []

    # Process each row
    for index, row in pandas_df.iterrows():
        try:
            # Create text content for embedding
            text_content = create_text_content(row.to_dict())

            if text_content:
                # Generate embeddings
                embeddings = generate_embeddings(text_content)

                if embeddings:
                    # Extract metadata for filtering in vector search
                    metadata = {
                        "asin": row.get('asin', ''),
                        "brand": row.get('brand', ''),
                        "seller_name": row.get('seller_name', ''),
                        "categories": row.get('categories', ''),
                        "department": row.get('department', ''),
                        "rating": float(row.get('rating', 0)) if row.get('rating') else 0,
                        "reviews_count": int(row.get('reviews_count', 0)) if row.get('reviews_count') else 0,
                        "final_price": float(row.get('final_price', 0)) if row.get('final_price') else 0,
                        "currency": row.get('currency', 'USD'),
                        "availability": row.get('availability', ''),
                        "discount": row.get('discount', ''),
                        "is_available": row.get('is_available', False),
                        "bought_past_month": int(row.get('bought_past_month', 0)) if row.get('bought_past_month') else 0,
                        "timestamp": row.get('timestamp', ''),
                    }

                    # Create processed record
                    processed_record = {
                        "id": row.get('asin', f"record_{index}"),  # Use ASIN as ID if available
                        "title": row.get('title', ''),
                        "url": row.get('url', ''),
                        "image_url": row.get('image_url', ''),
                        "text_content": text_content,
                        "embeddings": embeddings,
                        "embedding_dimension": len(embeddings),
                        "metadata": metadata,
                        "original_data": row.to_dict()
                    }
                    processed_records.append(processed_record)

                    if (index + 1) % 10 == 0:
                        print(f"Processed {index + 1} records...")

        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            continue

    print(f"Successfully processed {len(processed_records)} records with embeddings")

    # Save to vector store
    if processed_records:
        output_path = f"s3://{args['data-bucket']}/processed-data/"
        save_to_vector_store(processed_records, output_path)

    print("Data processing job completed successfully!")

if __name__ == "__main__":
    main()
    job.commit()