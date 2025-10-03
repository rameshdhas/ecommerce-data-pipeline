import json
import pandas as pd
import time
import os
from typing import List, Dict, Any
from shared.aws_clients import get_aws_clients



def save_to_s3_batch(processed_data: List[Dict[str, Any]], bucket: str, partition_id: str = None) -> bool:
    """
    Save processed data in optimized batches for better performance with large datasets
    """
    try:
        if not processed_data:
            return True

        aws_clients = get_aws_clients()

        # Create partition identifier for parallel processing
        worker_id = os.getpid()
        timestamp = pd.Timestamp.now()
        partition_suffix = f"_{partition_id}" if partition_id else f"_worker_{worker_id}_{int(time.time())}"

        # Smaller, more manageable JSON structure for large datasets
        output_data = {
            "timestamp": timestamp.isoformat(),
            "records_count": len(processed_data),
            "partition_info": {
                "worker_id": worker_id,
                "partition_id": partition_id
            },
            "records": processed_data
        }

        # Use date-based partitioning for better organization
        date_partition = timestamp.strftime('%Y/%m/%d')
        hour_partition = timestamp.strftime('%H')
        key = f"processed-data/{date_partition}/{hour_partition}/batch{partition_suffix}.json"

        print(f"Saving {len(processed_data)} records to s3://{bucket}/{key}")

        # Use streaming upload for large files
        json_content = json.dumps(output_data, separators=(',', ':'))  # Compact JSON

        aws_clients.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json_content,
            ContentType='application/json',
            Metadata={
                'records_count': str(len(processed_data)),
                'processing_timestamp': timestamp.isoformat(),
                'worker_id': str(worker_id)
            }
        )

        print(f"Successfully saved batch to S3: {len(processed_data)} records")
        return True

    except Exception as e:
        print(f"Error saving batch to S3: {str(e)}")
        return False