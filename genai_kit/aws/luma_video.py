import boto3
import json
from enum import Enum
from botocore.config import Config
from genai_kit.utils.random import seed
from genai_kit.aws.bedrock import BedrockModel


class LumaDuration(Enum):
    DURATION_5 = "5s"
    DURATION_9 = "9s"
    

class LumaSize(Enum):
    SIZE_1_1 = "1:1"
    SIZE_3_4 = "3:4"
    SIZE_4_3 = "4:3"
    SIZE_16_9 = "16:9"
    SIZE_9_16 = "9:16"
    SIZE_21_9 = "21:9"
    SIZE_9_21 = "9:21"


class BedrockLumaVideo():
    def __init__(self,
                 bucket_name: str,
                 region='us-west-2',
                 modelId: str = BedrockModel.LUMA_RAY2):
        self.bucket_name = bucket_name
        self.region = region
        self.modelId = modelId
        self.bedrock = boto3.client(
            service_name = 'bedrock-runtime',
            region_name = self.region,
            config = Config(
                connect_timeout=120,
                read_timeout=120,
                retries={'max_attempts': 5}
            ),
        )
        
    def generate_video(self, body: dict):
        invocation = self.bedrock.start_async_invoke(
            modelId=self.modelId,
            modelInput=body,
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{self.bucket_name}"
                }
            }
        )
        return invocation['invocationArn']


    def text_to_video(self,
                      prompt: str,
                      aspect_ratio: str=LumaSize.SIZE_16_9.value,
                      duration: str=LumaDuration.DURATION_9.value,
                      seed: int=seed(),
                      resolution: str='720p',
                      loop=True):
        body = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "seed": seed,
            "resolution": resolution,
            "loop": loop,
        }
        
        return self.generate_video(body=body)


    def image_to_video(self,
                        prompt: str,
                        imageBase64: str,
                        aspect_ratio: str=LumaSize.SIZE_16_9.value,
                        duration: str=LumaDuration.DURATION_9.value,
                        seed: int=seed(),
                        resolution: str='720p'):
        body = {
            "prompt": prompt,
            "mode": "image-to-image",
            "image": imageBase64,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "seed": seed,
            "resolution": resolution
        }
        
        return self.generate_video(body=body)

