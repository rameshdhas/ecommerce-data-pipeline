import boto3
from typing import Optional


class AWSClients:
    def __init__(self, bedrock_region: str = 'us-east-1'):
        self.bedrock_region = bedrock_region
        self._bedrock_client = None
        self._s3_client = None

    @property
    def bedrock_client(self):
        if self._bedrock_client is None:
            self._bedrock_client = boto3.client('bedrock-runtime', region_name=self.bedrock_region)
        return self._bedrock_client

    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client('s3')
        return self._s3_client


_aws_clients_instance: Optional[AWSClients] = None


def get_aws_clients(bedrock_region: str = 'us-east-1') -> AWSClients:
    global _aws_clients_instance
    if _aws_clients_instance is None:
        _aws_clients_instance = AWSClients(bedrock_region)
    return _aws_clients_instance