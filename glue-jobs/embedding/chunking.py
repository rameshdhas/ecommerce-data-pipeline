import json
from typing import List
from shared.aws_clients import get_aws_clients
from .models import get_max_chunk_size


def summarize_with_claude(text: str, model_id: str = "anthropic.claude-3-haiku-20240307") -> str:
    """
    Use Claude to summarize long text before generating embeddings.
    Claude models have 200K token context window.
    """
    try:
        aws_clients = get_aws_clients()

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

        response = aws_clients.bedrock_client.invoke_model(
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
        aws_clients = get_aws_clients()
        max_chunk_size = get_max_chunk_size(model_id)

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

            response = aws_clients.bedrock_client.invoke_model(
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