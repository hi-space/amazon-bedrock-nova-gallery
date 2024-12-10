
import boto3
from typing import Dict, Any, BinaryIO, List, Optional
from datetime import datetime
from genai_kit.aws.bedrock import BedrockModel
from genai_kit.aws.dynamodb import DynamoDB
from genai_kit.utils.random import random_id
from services.bedrock_service import list_video_job
from utils import extract_key_from_uri
from enums import MediaType
from config import config
from constants import IMAGE_PREFIX, VIDEO_OUTPUT_FILE, VIDEO_PREFIX


class StorageService:
    def __init__(self, bucket_name: str, cloudfront_domain: str):
        self.s3_client = boto3.client('s3')
        self.dynamodb = DynamoDB(table_name=config.DYNAMO_TABLE)
        self.bucket_name = bucket_name
        self.cloudfront_domain = cloudfront_domain
        
    def upload_media(
        self,
        media_type: MediaType,
        model_type: BedrockModel,
        prompt: str,
        details: Optional[Dict[str, Any]] = None,
        media_file: Optional[BinaryIO] = None,
        ref_image: Optional[str] = None,
    ):
        if media_type == MediaType.IMAGE:
            return self.upload_image(
                model_type=model_type.value,
                prompt=prompt,
                details=details,
                media_file=media_file,
                ref_image=ref_image,
            )
        elif media_type == MediaType.VIDEO:
            s3Uri = details.get("outputDataConfig", {}).get("s3OutputDataConfig", {}).get("s3Uri", "")
            id = extract_key_from_uri(s3Uri)
            return self.update_video_status(
                model_type=model_type.value,
                prompt=prompt,
                details=details,
                ref_image=ref_image,
                id=id,
            )    

    def upload_image(
        self,
        model_type: str,
        prompt: str,
        details: Dict[str, Any],
        media_file: Optional[BinaryIO] = None,
        ref_image: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        image_id = id or random_id()
        key = f"{IMAGE_PREFIX}/{image_id}"
        now = datetime.now().isoformat()
        url = None

        if media_file:
            url = self.upload_to_s3(media_file, key)

        url = url or f"{self.cloudfront_domain}/{key}"
        record = {
            "id": image_id,
            "media_type": MediaType.IMAGE.value,
            "model_type": model_type,
            "prompt": prompt,
            "ref_image": ref_image,
            "url": url,
            "updated_at": now,
            "details": details
        }

        try:
            existing_item = self.dynamodb.get_item(image_id)
            
            if existing_item:
                updates = {
                    'model_type': model_type,
                    'url': record['url'],
                    'updated_at': now,
                    'details': details
                }
                
                self.dynamodb.update_item(image_id, updates)
                record = self.dynamodb.get_item(image_id)
            else:
                record['created_at'] = now
                self.dynamodb.put_item(record)

        except Exception as e:
            raise Exception(f"Failed to store metadata in DynamoDB: {str(e)}")

        return record
    
    def update_video_status(
        self,
        model_type: Optional[str] = None,
        prompt: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ref_image: Optional[str] = None,
        id: Optional[str] = None,
    ) -> Dict[str, Any]:
        id = id or random_id()
        key = f"{VIDEO_PREFIX}/{id}"
        now = datetime.now().isoformat()
        
        record = {
            "id": id,
            "media_type": MediaType.VIDEO.value,
            "model_type": model_type,
            "prompt": prompt,
            "ref_image": ref_image,
            "url": f"{self.cloudfront_domain}/{key}/{VIDEO_OUTPUT_FILE}",
            "updated_at": now,
            "details": details
        }

        try:
            existing_item = self.dynamodb.get_item(id)
            
            if existing_item:
                updates = {
                    'url': record['url'],
                    'updated_at': now,
                    'details': details
                }
                
                self.dynamodb.update_item(id, updates)
                record = self.dynamodb.get_item(id)
            else:
                record['created_at'] = now
                self.dynamodb.put_item(record)

        except Exception as e:
            raise Exception(f"Failed to store metadata in DynamoDB: {str(e)}")

        return record

    def get_media_list(
        self,
        media_type: str = None,
        sync=True,
    ) -> List[Dict[str, Any]]:
        try:
            if sync:
                self.sync_video_jobs()

            if media_type:
                query = {
                    'FilterExpression': '#type = :type_val',
                    'ExpressionAttributeNames': {'#type': 'media_type'},
                    'ExpressionAttributeValues': {':type_val': media_type}
                }
                response = self.dynamodb.scan_items(query)
            else:
                response = self.dynamodb.scan_items({})
                
            return response.get('Items', [])
        except Exception as e:
            raise Exception(f"Failed to retrieve media list: {str(e)}")
        
    def sync_video_jobs(self) -> None:
        try:
            job_list = list_video_job(max_results=10)
            media_list = self.get_media_list(media_type=MediaType.VIDEO.value, sync=False)
            media_map = {item['id']: item for item in media_list}
        
            for job in job_list:
                job_id = extract_key_from_uri(
                    job.get('outputDataConfig', {}).get('s3OutputDataConfig', {}).get('s3Uri', '')
                )
                if not job_id or job_id not in media_map:
                    continue
                
                job_status = job.get('status')
                if media_map[job_id].get('details', {}).get('status', '') != job_status:
                    self.update_video_status(
                        details=job,
                        id=job_id
                    )

        except Exception as e:
            raise Exception(f"Failed to sync video jobs: {str(e)}")
        
    def upload_to_s3(self, image: BinaryIO, image_id: str) -> str:
        filename = f"{image_id}.png"
        
        try:
            self.s3_client.upload_fileobj(
                image,
                self.bucket_name,
                filename,
                ExtraArgs={'ContentType': 'image/png'}
            )
            return f"{self.cloudfront_domain}/{filename}"
        except Exception as e:
            raise Exception(f"Failed to upload image to S3: {str(e)}")
        
