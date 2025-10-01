import sys
import boto3
import pandas as pd
import json
import requests
import os
from datetime import datetime
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import DataFrame
from typing import List, Dict, Any, Optional

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Get required job arguments
required_args = ['JOB_NAME', 'data_bucket']
optional_args = ['input_path', 'batch_mode', 'elasticsearch_endpoint', 'elasticsearch_api_key', 'embedding_model']

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
    Generate vector embeddings using Amazon Bedrock foundation models.

    Available Bedrock embedding models:
    - amazon.titan-embed-text-v2: 8192 token limit, configurable 256/512/1024 dimensions
    - amazon.titan-embed-text-v1: 8192 token limit, 1536 dimensions
    - amazon.titan-embed-image-v1: For multimodal embeddings (1024 dimensions)
    - cohere.embed-english-v3: 512 token limit, 1024 dimensions
    - cohere.embed-multilingual-v3: 512 token limit, 1024 dimensions

    For long texts that exceed token limits:
    - Use Claude for intelligent summarization before embedding
    - Or use automatic chunking with embedding averaging
    """
    try:
        # Claude models via Bedrock for text summarization + embedding
        # Claude doesn't generate embeddings directly, but can summarize long text
        if model_id.startswith("anthropic.claude"):
            # Claude doesn't provide embeddings directly
            # We'll use it to summarize long text, then use Titan for embeddings
            print(f"Claude models don't provide embeddings. Using Claude to summarize, then Titan for embeddings.")

            # First summarize with Claude if text is too long
            if len(text) > 20000:
                summarized_text = summarize_with_claude(text, model_id)
                # Generate embeddings with Titan on the summarized text
                return generate_embeddings(summarized_text, "amazon.titan-embed-text-v1")
            else:
                # If text is short enough, just use Titan directly
                return generate_embeddings(text, "amazon.titan-embed-text-v1")

        # Amazon Titan models
        elif model_id == "amazon.titan-embed-text-v2":
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
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

        elif model_id == "amazon.titan-embed-text-v1":
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

        # Amazon Titan Image Embedding (for multimodal content)
        elif model_id == "amazon.titan-embed-image-v1":
            # This model can handle both text and images
            body = json.dumps({
                "inputText": text,
                "embeddingConfig": {
                    "outputEmbeddingLength": 1024
                }
            })

            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

        # Cohere models via Bedrock
        elif model_id.startswith("cohere.embed"):
            body = json.dumps({
                "texts": [text],
                "input_type": "search_document",
                "embedding_types": ["float"]
            })

            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embeddings']['float'][0]

        else:
            # Default to Titan v2
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })

            response = bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

    except Exception as e:
        error_msg = str(e)
        print(f"Error generating embeddings: {error_msg}")
        print(f"  - Model ID: {model_id}")
        print(f"  - Text length: {len(text)} characters")
        print(f"  - Text preview: {text[:100]}..." if len(text) > 100 else f"  - Text: {text}")

        # If model is invalid, try fallback to v1
        if "ValidationException" in error_msg and "invalid" in error_msg.lower():
            if model_id == "amazon.titan-embed-text-v2":
                print("Titan v2 model not available, falling back to Titan v1...")
                return generate_embeddings(text, "amazon.titan-embed-text-v1")
            elif "amazon.titan-embed" not in model_id:
                print("Model not available, falling back to Titan v1...")
                return generate_embeddings(text, "amazon.titan-embed-text-v1")

        # If token limit exceeded, try chunking the text
        if "Too many input tokens" in error_msg or "token" in error_msg.lower():
            print("Token limit exceeded. Attempting to chunk and summarize text...")
            return generate_embeddings_with_chunking(text, model_id)

        return []

def summarize_with_claude(text: str, model_id: str = "anthropic.claude-3-haiku-20240307") -> str:
    """
    Use Claude to summarize long text before generating embeddings.
    Claude models have 200K token context window.
    """
    try:
        # Prepare the prompt for summarization
        prompt = f"""Please summarize the following product information, keeping all important details for search and discovery.
        Focus on: product name, brand, key features, categories, and specifications.
        Keep the summary under 5000 characters.

        Product Information:
        {text}

        Summary:"""

        # Prepare the request body for Claude
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1
        })

        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())
        summarized_text = response_body['content'][0]['text']

        print(f"Text summarized from {len(text)} to {len(summarized_text)} characters using Claude")
        return summarized_text

    except Exception as e:
        print(f"Error summarizing with Claude: {str(e)}")
        # Fall back to truncation if summarization fails
        return text[:5000]

def generate_embeddings_with_chunking(text: str, model_id: str) -> List[float]:
    """
    Generate embeddings for long text by chunking it into smaller pieces.
    This method chunks the text and averages the embeddings.
    """
    try:
        # Chunk size based on model (conservative estimates)
        if model_id in ["amazon.titan-embed-text-v2", "amazon.titan-embed-text-v1"]:
            max_chunk_size = 20000  # ~8000 tokens for Titan models
        elif model_id == "amazon.titan-embed-image-v1":
            max_chunk_size = 20000  # ~8000 tokens
        elif model_id.startswith("anthropic.claude"):
            # Claude has 200K context, use larger chunks for summarization
            max_chunk_size = 100000  # Can handle much larger chunks
        elif model_id.startswith("cohere"):
            max_chunk_size = 1200  # ~500 tokens for Cohere
        else:
            max_chunk_size = 20000  # Default to Titan-like limits

        # Split text into chunks
        chunks = []
        for i in range(0, len(text), max_chunk_size):
            chunk = text[i:i + max_chunk_size]
            chunks.append(chunk)

        print(f"Text split into {len(chunks)} chunks")

        # Generate embeddings for each chunk
        all_embeddings = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")

            if model_id == "amazon.titan-embed-text-v2":
                body = json.dumps({
                    "inputText": chunk,
                    "dimensions": 1024,
                    "normalize": True
                })
            elif model_id == "amazon.titan-embed-text-v1":
                body = json.dumps({
                    "inputText": chunk
                })
            elif model_id.startswith("cohere"):
                body = json.dumps({
                    "texts": [chunk],
                    "input_type": "search_document",
                    "embedding_types": ["float"]
                })
            else:
                body = json.dumps({
                    "inputText": chunk,
                    "dimensions": 1024,
                    "normalize": True
                })

            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())

            if model_id.startswith("cohere"):
                embedding = response_body['embeddings']['float'][0]
            else:
                embedding = response_body['embedding']

            all_embeddings.append(embedding)

        # Average the embeddings
        if all_embeddings:
            try:
                import numpy as np
            except ImportError:
                print("Warning: numpy not available, using manual averaging")
                # Manual averaging fallback
                if all_embeddings:
                    num_embeddings = len(all_embeddings)
                    embedding_length = len(all_embeddings[0])
                    averaged = []
                    for i in range(embedding_length):
                        avg_val = sum(emb[i] for emb in all_embeddings) / num_embeddings
                        averaged.append(avg_val)
                    return averaged
                return []
            averaged = np.mean(all_embeddings, axis=0).tolist()
            print(f"Successfully generated averaged embedding from {len(chunks)} chunks")
            return averaged

        return []

    except Exception as e:
        print(f"Error in chunked embedding generation: {str(e)}")
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

def get_embedding_dimensions(model_id: str) -> int:
    """
    Get the embedding dimensions for a given Bedrock model.
    """
    if model_id == "amazon.titan-embed-text-v1":
        return 1536
    elif model_id in ["amazon.titan-embed-text-v2", "amazon.titan-embed-image-v1"]:
        return 1024  # Using 1024 for Titan v2 (configurable: 256/512/1024)
    elif model_id.startswith("cohere.embed"):
        return 1024  # Cohere v3 models
    elif model_id.startswith("anthropic.claude"):
        # Claude doesn't generate embeddings directly, uses Titan for embeddings
        return 1024  # Will use Titan v2 embeddings after summarization
    else:
        return 1024  # Default

def save_to_elasticsearch(processed_data: List[Dict[str, Any]]):
    """
    Save processed data with embeddings to Elasticsearch
    """
    # Get Elasticsearch configuration
    es_endpoint = args.get('elasticsearch_endpoint', '')
    es_api_key = args.get('elasticsearch_api_key', '')

    if not es_endpoint or not es_api_key:
        print("Warning: Elasticsearch endpoint or API key not provided. Skipping Elasticsearch indexing.")
        return False

    # Get the embedding model to determine dimensions
    embedding_model = args.get('embedding_model', 'amazon.titan-embed-text-v1')
    embedding_dims = get_embedding_dimensions(embedding_model)

    # Create index name with timestamp
    index_name = f"ecommerce-products"

    # Elasticsearch headers
    headers = {
        'Authorization': f'ApiKey {es_api_key}',
        'Content-Type': 'application/json'
    }

    try:
        # First, check if index exists and create it if not
        index_exists_url = f"{es_endpoint}/{index_name}"
        response = requests.head(index_exists_url, headers=headers)

        if response.status_code == 404:
            # Create index with mapping for vector search
            print(f"Creating Elasticsearch index: {index_name}")
            print(f"Using embedding dimensions: {embedding_dims} for model: {embedding_model}")

            index_mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "url": {"type": "text"},
                        "image_url": {"type": "text"},
                        "text_content": {"type": "text"},
                        "embeddings": {
                            "type": "dense_vector",
                            "dims": embedding_dims,  # Dynamic dimensions based on model
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "properties": {
                                "asin": {"type": "keyword"},
                                "brand": {"type": "keyword"},
                                "seller_name": {"type": "keyword"},
                                "categories": {"type": "text"},
                                "department": {"type": "keyword"},
                                "rating": {"type": "float"},
                                "reviews_count": {"type": "integer"},
                                "final_price": {"type": "float"},
                                "currency": {"type": "keyword"},
                                "availability": {"type": "text"},
                                "discount": {"type": "text"},
                                "is_available": {"type": "boolean"},
                                "bought_past_month": {"type": "integer"},
                                "timestamp": {"type": "date"}
                            }
                        },
                        "indexed_at": {"type": "date"}
                    }
                }
            }

            create_response = requests.put(
                index_exists_url,
                headers=headers,
                json=index_mapping
            )

            if create_response.status_code not in [200, 201]:
                print(f"Error creating index: {create_response.text}")
                return False
            else:
                print(f"Successfully created index: {index_name}")

        # Bulk index documents
        bulk_url = f"{es_endpoint}/_bulk"
        bulk_data = []

        print(f"Total processed records to check: {len(processed_data)}")
        records_with_embeddings = 0
        records_without_embeddings = 0

        for record in processed_data:
            # Only index records that have embeddings (non-empty list)
            embeddings = record.get('embeddings', [])
            if embeddings and len(embeddings) > 0:
                records_with_embeddings += 1
                # Prepare document for Elasticsearch
                doc = {
                    "id": record['id'],
                    "title": record['title'],
                    "url": record['url'],
                    "image_url": record['image_url'],
                    "text_content": record['text_content'],
                    "embeddings": record['embeddings'],
                    "metadata": record['metadata'],
                    "indexed_at": datetime.now().isoformat()
                }

                # Add bulk index action
                bulk_data.append(json.dumps({"index": {"_index": index_name, "_id": record['id']}}))
                bulk_data.append(json.dumps(doc))
            else:
                records_without_embeddings += 1
                print(f"Skipping record {record.get('id', 'unknown')}: No embeddings or empty embeddings list")
                print(f"  - has_embeddings: {record.get('has_embeddings', False)}")
                print(f"  - embeddings length: {len(record.get('embeddings', []))}")

        print(f"Records with embeddings: {records_with_embeddings}")
        print(f"Records without embeddings: {records_without_embeddings}")

        if bulk_data:
            # Join with newlines and add final newline
            bulk_body = '\n'.join(bulk_data) + '\n'

            print(f"Indexing {len(bulk_data)//2} documents to Elasticsearch...")
            print(f"Bulk request size: {len(bulk_body)} bytes")
            print(f"First bulk action: {bulk_data[0] if bulk_data else 'No data'}")
            print(f"First document: {bulk_data[1] if len(bulk_data) > 1 else 'No document'}")

            response = requests.post(
                bulk_url,
                headers=headers,
                data=bulk_body
            )

            print(f"Elasticsearch response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Bulk response - Errors: {result.get('errors', 'N/A')}, Took: {result.get('took', 'N/A')}ms")

                # Count successful and failed indexing
                success_count = 0
                error_count = 0

                for item in result.get('items', []):
                    if 'index' in item:
                        if item['index'].get('status') in [200, 201]:
                            success_count += 1
                        else:
                            error_count += 1
                            if 'error' in item['index']:
                                error_detail = item['index']['error']
                                print(f"Error indexing document {item['index'].get('_id', 'unknown')}: {error_detail}")

                print(f"Indexing results - Success: {success_count}, Errors: {error_count}")

                if not result.get('errors', True):
                    print(f"Successfully indexed {len(bulk_data)//2} documents to Elasticsearch index: {index_name}")
                    return True
                else:
                    print(f"Bulk indexing completed with errors. Check logs above for details.")
                    return False
            else:
                print(f"Error during bulk indexing: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response body: {response.text[:1000]}")  # First 1000 chars
                return False
        else:
            print("No documents with embeddings to index to Elasticsearch")
            return False

    except Exception as e:
        print(f"Error saving to Elasticsearch: {str(e)}")
        return False

def save_to_s3_and_elasticsearch(processed_data: List[Dict[str, Any]]):
    """
    Save processed data with embeddings to both S3 and Elasticsearch
    """
    # Save to S3
    try:
        # Convert to JSON for S3 storage
        output_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data": processed_data,
            "total_records": len(processed_data)
        }

        # Save to S3 as JSON
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

    except Exception as e:
        print(f"Error saving to S3: {str(e)}")
        print(f"Bucket: {bucket}, Key: {key}")
        # Continue even if S3 save fails

    # Save to Elasticsearch
    es_success = save_to_elasticsearch(processed_data)

    if es_success:
        print("Data successfully saved to both S3 and Elasticsearch")
    else:
        print("Data saved to S3, but Elasticsearch indexing failed or was skipped")

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

    # Get the embedding model to use (default to Titan v1 if not specified)
    embedding_model = args.get('embedding_model', 'amazon.titan-embed-text-v1')
    print(f"Using embedding model: {embedding_model}")

    # Process each row
    for index, row in pandas_df.iterrows():
        try:
            # Create text content for embedding
            text_content = create_text_content(row.to_dict())

            if text_content:
                # Generate embeddings using the specified model
                embeddings = generate_embeddings(text_content, model_id=embedding_model)

                # For now, continue even if embeddings fail (store record without embeddings)
                # This allows us to at least save the processed metadata
                if embeddings or True:  # Always process records even without embeddings
                    # Extract metadata for filtering in vector search
                    # Convert timestamp to ISO format for Elasticsearch
                    timestamp_value = row.get('timestamp', '')
                    if timestamp_value:
                        try:
                            # Parse the timestamp and convert to ISO format
                            if isinstance(timestamp_value, str):
                                # Handle format like "2023-08-08 00:00:00.000"
                                dt = pd.to_datetime(timestamp_value)
                                timestamp_iso = dt.isoformat()
                            else:
                                timestamp_iso = pd.to_datetime(timestamp_value).isoformat()
                        except Exception as e:
                            print(f"Warning: Could not parse timestamp '{timestamp_value}': {e}")
                            timestamp_iso = datetime.now().isoformat()
                    else:
                        timestamp_iso = datetime.now().isoformat()

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
                        "timestamp": timestamp_iso,
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

    # Save to S3 and Elasticsearch
    if processed_records:
        save_to_s3_and_elasticsearch(processed_records)
        print(f"Processing complete. Records with embeddings: {successful_embeddings}/{len(processed_records)}")
    else:
        print("WARNING: No records were processed!")

    print("Data processing job completed successfully!")

if __name__ == "__main__":
    main()
    job.commit()