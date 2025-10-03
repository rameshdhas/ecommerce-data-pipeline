import json
import requests
from datetime import datetime
from typing import List, Dict, Any
from embedding.models import get_embedding_dimensions




def save_to_elasticsearch_batch(processed_data: List[Dict[str, Any]], es_endpoint: str, es_api_key: str, embedding_model: str) -> bool:
    """
    Optimized batch saving for large-scale Elasticsearch operations
    """
    if not es_endpoint or not es_api_key:
        print("Warning: Elasticsearch endpoint or API key not provided. Skipping Elasticsearch indexing.")
        return False

    if not processed_data:
        return True

    # Get the embedding model to determine dimensions
    embedding_dims = get_embedding_dimensions(embedding_model)
    index_name = f"ecommerce-products"

    headers = {
        'Authorization': f'ApiKey {es_api_key}',
        'Content-Type': 'application/json'
    }

    try:
        # Filter records with embeddings early
        records_with_embeddings = [
            record for record in processed_data
            if record.get('embeddings') and len(record.get('embeddings', [])) > 0
        ]

        if not records_with_embeddings:
            print(f"No records with embeddings in this batch ({len(processed_data)} total records)")
            return True

        print(f"Batch processing {len(records_with_embeddings)} records with embeddings (out of {len(processed_data)} total)")

        # Optimized bulk indexing for large batches
        bulk_url = f"{es_endpoint}/_bulk"
        batch_size = 100  # Larger batch size for better performance

        # Prepare all documents efficiently
        documents = []
        for record in records_with_embeddings:
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
            documents.append(doc)

        # Process in optimized batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        total_success = 0

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]

            # Create bulk request efficiently
            bulk_lines = []
            for doc in batch_docs:
                bulk_lines.append(json.dumps({"index": {"_index": index_name, "_id": doc['id']}}))
                bulk_lines.append(json.dumps(doc))

            bulk_body = '\n'.join(bulk_lines) + '\n'

            try:
                response = requests.post(
                    bulk_url,
                    headers=headers,
                    data=bulk_body,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    # Count successes efficiently
                    batch_success = sum(
                        1 for item in result.get('items', [])
                        if 'index' in item and item['index'].get('status') in [200, 201]
                    )

                    total_success += batch_success

                    if batch_num % 5 == 0:  # Log every 5th batch
                        print(f"Processed batch {batch_num + 1}/{total_batches}: {batch_success}/{len(batch_docs)} successful")

                else:
                    print(f"Batch {batch_num + 1} failed with status {response.status_code}")

            except Exception as e:
                print(f"Error in batch {batch_num + 1}: {str(e)}")
                continue

        success_rate = total_success / len(documents) if documents else 0
        print(f"Elasticsearch batch indexing: {total_success}/{len(documents)} successful ({success_rate:.1%})")

        return success_rate >= 0.8  # 80% success threshold for batch operations

    except Exception as e:
        print(f"Error in batch Elasticsearch operation: {str(e)}")
        return False