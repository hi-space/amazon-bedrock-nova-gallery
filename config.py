import boto3
import json
import os
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()

class Settings(BaseModel):
    BEDROCK_REGION: str
    DYNAMO_TABLE: str
    S3_BUCKET: str
    CF_DOMAIN: str


def get_secrets_from_manager():
    SECRET_NAME = "bedrock-gallery"
    try:
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager')
        get_secret_value_response = client.get_secret_value(SecretId=SECRET_NAME)
        secrets = json.loads(get_secret_value_response['SecretString'])
        return Settings(**secrets)
    except Exception as e:
        return None


def get_settings():
    settings = get_secrets_from_manager()

    if settings is None:
        try:
            settings = Settings(
                BEDROCK_REGION=os.getenv("BEDROCK_REGION"),
                DYNAMO_TABLE=os.getenv("DYNAMO_TABLE"),
                S3_BUCKET=os.getenv("S3_BUCKET"),
                CF_DOMAIN=os.getenv("CF_DOMAIN"),
            )
        except Exception as e:
            return None

    return settings

config = get_settings()