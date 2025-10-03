import json
import traceback
from typing import List
from shared.aws_clients import get_aws_clients
from .chunking import generate_embeddings_with_chunking, summarize_with_claude


def generate_embeddings(text: str, model_id: str = "amazon.titan-embed-text-v1") -> List[float]:
    """
    Generate vector embeddings using Amazon Bedrock foundation models.
    For long texts that exceed token limits:
    - Use Claude for intelligent summarization before embedding
    - Or use automatic chunking with embedding averaging
    """
    try:
        print(f"DEBUG: Starting embedding generation for text length {len(text)}")
        aws_clients = get_aws_clients()
        print(f"DEBUG: Got AWS clients successfully")

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
        elif model_id == "amazon.titan-embed-text-v1":
            print(f"DEBUG: Using Titan v1 model for embedding")
            body = json.dumps({
                "inputText": text
            })
            print(f"DEBUG: Prepared request body")

            response = aws_clients.bedrock_client.invoke_model(
                modelId=model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            print(f"DEBUG: Got response from Bedrock")

            response_body = json.loads(response['body'].read())
            embeddings = response_body['embedding']
            print(f"DEBUG: Successfully extracted {len(embeddings)} embedding dimensions")
            return embeddings
        else:
            # Default to Titan v2
            body = json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })

            response = aws_clients.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=body,
                contentType='application/json',
                accept='application/json'
            )

            response_body = json.loads(response['body'].read())
            return response_body['embedding']

    except Exception as e:
        error_msg = str(e)
        print(f"ERROR generating embeddings: {error_msg}")
        print(f"ERROR type: {type(e).__name__}")
        print(f"  - Model ID: {model_id}")
        print(f"  - Text length: {len(text)} characters")
        print(f"  - Text preview: {text[:100]}..." if len(text) > 100 else f"  - Text: {text}")

        # Print full traceback for debugging
        print(f"ERROR traceback: {traceback.format_exc()}")

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


