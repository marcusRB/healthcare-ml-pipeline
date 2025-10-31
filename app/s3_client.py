import boto3
import pandas as pd
from io import StringIO, BytesIO
from botocore.exceptions import ClientError

class S3Client:
    def __init__(self):
        # Uses IAM role attached to EC2 instance or ECS task
        self.client = boto3.client('s3')
    
    def read_csv(self, bucket: str, key: str) -> pd.DataFrame:
        """Read CSV file from S3 and return as DataFrame"""
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            csv_content = response['Body'].read().decode('utf-8')
            return pd.read_csv(StringIO(csv_content))
        except ClientError as e:
            print(f"Error reading CSV file: {e}")
            raise
    
    def write_csv(self, df: pd.DataFrame, bucket: str, key: str):
        """Write DataFrame to CSV file in S3"""
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )
        except ClientError as e:
            print(f"Error writing CSV file: {e}")
            raise
    
    def read_json(self, bucket: str, key: str) -> dict:
        """Read JSON file from S3"""
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            return pd.read_json(StringIO(content))
        except ClientError as e:
            print(f"Error reading JSON file: {e}")
            raise
    
    def write_json(self, data: dict, bucket: str, key: str):
        """Write JSON data to S3"""
        try:
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=pd.io.json.dumps(data).encode('utf-8'),
                ContentType='application/json'
            )
        except ClientError as e:
            print(f"Error writing JSON file: {e}")
            raise
    
    def list_files(self, bucket: str, prefix: str = ""):
        """List files in S3 bucket with prefix"""
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            print(f"Error listing files: {e}")
            raise