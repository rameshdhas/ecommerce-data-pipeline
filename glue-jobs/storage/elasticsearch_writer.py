import json
import requests
from datetime import datetime
from typing import List, Dict, Any
from embedding.models import get_embedding_dimensions


def save_to_elasticsearch(processed_data: List[Dict[str, Any]], es_endpoint: str, es_api_key: str, embedding_model: str) -> bool:
    """
    Save processed data with embeddings to Elasticsearch
    """
    if not es_endpoint or not es_api_key:
        print("Warning: Elasticsearch endpoint or API key not provided. Skipping Elasticsearch indexing.")
        return False

    # Get the embedding model to determine dimensions
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
                        "description": {"type": "text"},
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

        # Bulk index documents with batching to handle large payloads
        bulk_url = f"{es_endpoint}/_bulk"

        print(f"Total processed records to check: {len(processed_data)}")
        records_with_embeddings = 0
        records_without_embeddings = 0

        # Prepare documents for indexing
        documents_to_index = []

        for record in processed_data:
            # Only index records that have embeddings (non-empty list)
            embeddings = record.get('embeddings', [])
            if embeddings and len(embeddings) > 0:
                records_with_embeddings += 1
                # Prepare document for Elasticsearch
                doc = {
                    "id": record['id'],
                    "title": record['title'],
                    "description": record.get('description', ''),
                    "url": record['url'],
                    "image_url": record['image_url'],
                    "text_content": record['text_content'],
                    "embeddings": record['embeddings'],
                    "metadata": record['metadata'],
                    "indexed_at": datetime.now().isoformat()
                }
                documents_to_index.append(doc)
            else:
                records_without_embeddings += 1
                print(f"Skipping record {record.get('id', 'unknown')}: No embeddings or empty embeddings list")
                print(f"  - has_embeddings: {record.get('has_embeddings', False)}")
                print(f"  - embeddings length: {len(record.get('embeddings', []))}")

        print(f"Records with embeddings: {records_with_embeddings}")
        print(f"Records without embeddings: {records_without_embeddings}")

        if not documents_to_index:
            print("No documents with embeddings to index to Elasticsearch")
            return False

        # Batch size for bulk indexing - smaller batches to avoid payload size issues
        batch_size = 50  # Process 50 documents per batch to keep request size manageable
        total_batches = (len(documents_to_index) + batch_size - 1) // batch_size

        print(f"Indexing {len(documents_to_index)} documents in {total_batches} batches of {batch_size} documents each...")

        total_success_count = 0
        total_error_count = 0

        # Process documents in batches
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(documents_to_index))
            batch_docs = documents_to_index[start_idx:end_idx]

            # Prepare bulk data for this batch
            bulk_data = []
            for doc in batch_docs:
                bulk_data.append(json.dumps({"index": {"_index": index_name, "_id": doc['id']}}))
                bulk_data.append(json.dumps(doc))

            # Join with newlines and add final newline
            bulk_body = '\n'.join(bulk_data) + '\n'

            print(f"Processing batch {batch_num + 1}/{total_batches}: {len(batch_docs)} documents, {len(bulk_body)} bytes")

            try:
                response = requests.post(
                    bulk_url,
                    headers=headers,
                    data=bulk_body,
                    timeout=30  # Add timeout to prevent hanging
                )

                print(f"Batch {batch_num + 1} response status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    print(f"Batch {batch_num + 1} - Errors: {result.get('errors', 'N/A')}, Took: {result.get('took', 'N/A')}ms")

                    # Count successful and failed indexing for this batch
                    batch_success_count = 0
                    batch_error_count = 0

                    for item in result.get('items', []):
                        if 'index' in item:
                            if item['index'].get('status') in [200, 201]:
                                batch_success_count += 1
                            else:
                                batch_error_count += 1
                                if 'error' in item['index']:
                                    error_detail = item['index']['error']
                                    print(f"Error indexing document {item['index'].get('_id', 'unknown')}: {error_detail}")

                    total_success_count += batch_success_count
                    total_error_count += batch_error_count

                    print(f"Batch {batch_num + 1} results - Success: {batch_success_count}, Errors: {batch_error_count}")

                else:
                    print(f"Error in batch {batch_num + 1}: {response.status_code}")
                    print(f"Response headers: {response.headers}")
                    print(f"Response body: {response.text[:500]}")
                    total_error_count += len(batch_docs)

            except Exception as e:
                print(f"Exception during batch {batch_num + 1} indexing: {str(e)}")
                total_error_count += len(batch_docs)
                continue

        print(f"Final indexing results - Total Success: {total_success_count}, Total Errors: {total_error_count}")

        # Return success if most documents were indexed successfully
        success_rate = total_success_count / len(documents_to_index) if documents_to_index else 0
        if success_rate >= 0.9:  # 90% success rate threshold
            print(f"Successfully indexed {total_success_count}/{len(documents_to_index)} documents to Elasticsearch index: {index_name}")
            return True
        else:
            print(f"Bulk indexing completed with significant errors. Success rate: {success_rate:.2%}")
            return False

    except Exception as e:
        print(f"Error saving to Elasticsearch: {str(e)}")
        return False