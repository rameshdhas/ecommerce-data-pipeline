def get_embedding_dimensions(model_id: str) -> int:
    """
    Get the embedding dimensions for a given Bedrock model.
    """
    if model_id == "amazon.titan-embed-text-v1":
        return 1536
    else:
        return 1024  # Default


def get_max_chunk_size(model_id: str) -> int:
    """
    Get the maximum chunk size for a given model.
    """
    if model_id in ["amazon.titan-embed-text-v2", "amazon.titan-embed-text-v1"]:
        return 20000  # ~8000 tokens for Titan models
    elif model_id.startswith("anthropic.claude"):
        # Claude has 200K context, use larger chunks for summarization
        return 100000  # Can handle much larger chunks
    else:
        return 20000  # Default to Titan-like limits