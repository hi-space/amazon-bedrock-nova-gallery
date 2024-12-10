import os
import boto3
import datetime
from urllib.parse import urlparse, quote, unquote


class S3:
    def __init__(self, bucket_name, region='us-west-2'):
        self.storage = boto3.client('s3',                            
                                    region_name = region)
        self.bucket_name = bucket_name

    def upload_object(self, bytes, key, metadata=None, extra_args=None):
        extra_args = extra_args or {}
        if metadata:
            formatted_metadata = {
                f'x-amz-meta-{k.lower()}': quote(str(v)) 
                for k, v in metadata.items()
            }
            extra_args['Metadata'] = formatted_metadata
        
        self.storage.upload_fileobj(
            bytes,
            self.bucket_name,
            key,
            ExtraArgs=extra_args
        )

    def get_object(self, key, include_metadata=False):
        response = self.storage.get_object(Bucket=self.bucket_name, Key=key)
        content = response['Body'].read()
        
        if include_metadata:
            metadata = {
                k.replace('x-amz-meta-', ''): unquote(v)
                for k, v in response.get('Metadata', {}).items()
            }
            return content, metadata
        return content

    def get_object_metadata(self, key):
        try:
            response = self.storage.head_object(Bucket=self.bucket_name, Key=key)
            return {
                k.replace('x-amz-meta-', ''): unquote(v)
                for k, v in response.get('Metadata', {}).items()
            }
        except Exception as e:
            print(f"Error getting metadata for {key}: {e}")
            return {}


    def download_object(self, key, file_path):
        self.storage.download_file(self.bucket_name, key, file_path)

    def extract_key_from_uri(self, s3_uri):
        parsed_uri = urlparse(s3_uri)
        return parsed_uri.path.lstrip('/')
    
    def list_objects(self, formats=None, prefix='', recursive=True, include_metadata=False):
        """
        특정 파일 포맷에 해당하는 S3 객체의 키 목록과 메타데이터를 최신 순으로 반환합니다.
        
        Args:
            formats (list): 검색할 파일 포맷 리스트 (예: ['.json', '.csv'])
            prefix (str): 검색할 경로 접두사
            recursive (bool): 하위 디렉토리까지 검색할지 여부
            include_metadata (bool): 메타데이터 포함 여부
        
        Returns:
            list: 최신 순으로 정렬된 파일 정보 목록 (메타데이터 포함 시 딕셔너리 형태)
        """
        object_info = []
        paginator = self.storage.get_paginator('list_objects_v2')
        
        formats = [fmt.lower() if not fmt.startswith('.') else fmt.lower() for fmt in formats] if formats else []
        
        kwargs = {
            'Bucket': self.bucket_name,
            'Prefix': prefix
        }
        if not recursive:
            kwargs['Delimiter'] = '/'
        
        try:
            for page in paginator.paginate(**kwargs):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if formats:
                            file_extension = os.path.splitext(key)[1].lower()
                            if file_extension not in formats:
                                continue
                        
                        if include_metadata:
                            metadata = self.get_object_metadata(key)
                            object_info.append({
                                'key': key,
                                'last_modified': obj['LastModified'],
                                'metadata': metadata
                            })
                        else:
                            object_info.append((key, obj['LastModified']))
                
                if not recursive and 'CommonPrefixes' in page:
                    for prefix_obj in page['CommonPrefixes']:
                        if include_metadata:
                            metadata = self.get_object_metadata(key)
                            object_info.append({
                                'key': prefix_obj['Prefix'],
                                'last_modified': datetime.datetime.now(datetime.timezone.utc),
                                'metadata': metadata
                            })
                        else:
                            object_info.append((prefix_obj['Prefix'], 
                                              datetime.datetime.now(datetime.timezone.utc)))
                        
        except Exception as e:
            print(f"Error listing objects: {e}")
            raise
        
        if include_metadata:
            return sorted(object_info, key=lambda x: x['last_modified'], reverse=True)
        return [item[0] for item in sorted(object_info, key=lambda x: x[1], reverse=True)]