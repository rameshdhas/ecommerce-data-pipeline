import json
import pandas as pd
from typing import List, Dict, Any
from shared.aws_clients import get_aws_clients


def save_to_s3(processed_data: List[Dict[str, Any]], bucket: str) -> bool:
    """
    Save processed data with embeddings to S3
    """
    try:
        aws_clients = get_aws_clients()

        # Convert to JSON for S3 storage
        output_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data": processed_data,
            "total_records": len(processed_data)
        }

        # Save to S3 as JSON
        key = f"processed-data/{pd.Timestamp.now().strftime('%Y/%m/%d')}/embeddings_{pd.Timestamp.now().strftime('%H%M%S')}.json"

        print(f"Attempting to save {len(processed_data)} records to s3://{bucket}/{key}")

        aws_clients.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(output_data, indent=2),
            ContentType='application/json'
        )

        print(f"Successfully saved processed data to s3://{bucket}/{key}")
        return True

    except Exception as e:
        print(f"Error saving to S3: {str(e)}")
        print(f"Bucket: {bucket}, Key: {key}")
        return False