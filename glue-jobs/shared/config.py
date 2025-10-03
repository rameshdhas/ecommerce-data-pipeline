import sys
from awsglue.utils import getResolvedOptions
from typing import Dict, Any


def get_job_arguments() -> Dict[str, Any]:
    """Parse and return job arguments"""
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

    # Debug: Print the arguments received
    print(f"Job initialized with arguments: {args}")
    print(f"Data bucket: {args.get('data_bucket', 'NOT FOUND')}")

    return args