import boto3
import json
from enum import Enum
from botocore.config import Config
from genai_kit.utils.random import seed


class BedrockStableDiffusion():
    def __init__(self, modelId: str, region='us-west-2'):
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

    def invoke_model(self, body: dict):
        response = self.bedrock.invoke_model(
            body=json.dumps(body),
            modelId=self.modelId
        )
        response_body = json.loads(response.get("body").read())
        return response_body["images"][0]

    def text_to_image(self,
                      prompt: str,
                      aspect_ratio: str='1:1',
                      seed: int=seed(),
                      format: str='png'):
        body = {
            "prompt": prompt,
            "mode": "text-to-image",
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "output_format": format
        }

        return self.invoke_model(body=body)

    def image_to_image(self,
                       prompt: str,
                       imageBase64: str,
                       strength: float=0.7,
                       seed: int=seed(),
                       format: str='png'):
        body = {
            "prompt": prompt,
            "mode": "image-to-image",
            "image": imageBase64,
            "strength": strength,
            "seed": seed,
            "output_format": format
        }

        return self.invoke_model(body=body)


class SDImageSize(Enum):
    SIZE_1_1 = "1:1"
    SIZE_16_9 = "16:9"
    SIZE_21_9 = "21:9"
    SIZE_2_3 = "2:3"
    SIZE_3_2 = "3:2"
    SIZE_4_5 = "4:5"
    SIZE_5_4 = "5:4"
    SIZE_9_16 = "9:16"
    SIZE_9_32 = "9:32"
