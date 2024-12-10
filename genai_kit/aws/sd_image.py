import boto3
import json
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
        try:
            response = self.bedrock.invoke_model(
                body=json.dumps(body),
                modelId=self.modelId
            )
            response_body = json.loads(response.get("body").read())
            return response_body["images"][0]
        except Exception as e:
            print(f"Cannot generate a image: {e}")
            return []

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
