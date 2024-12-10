import boto3
from enum import Enum


class BedrockModel(str, Enum):
    SONNET_3_5 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    SONNET_3_5_CR = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"

    HAIKU_3_5 = "anthropic.claude-3-5-haiku-20241022-v1:0"
    HAIKU_3_5_CR = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

    OPUS_3_0 = "anthropic.claude-3-opus-20240229-v1:0"

    TITAN_IMAGE = "amazon.titan-image-generator-v2:0"
    SD3_LARGE = "stability.sd3-large-v1:0"
    STABLE_IMAGE_CORE = "stability.stable-image-core-v1:0"
    STABLE_IMAGE_ULTRA = "stability.stable-image-ultra-v1:0"
    
    TITAN_TEXT_EMBEDDING = "amazon.titan-embed-text-v2:0"
    TITAN_MULTIMODAL_EMBEDDING = "amazon.titan-embed-image-v1"
    COHERE_MULTILINGUAL = "cohere.embed-multilingual-v3"

    NOVA_PRO = "amazon.nova-pro-v1:0"
    NOVA_LITE = "amazon.nova-lite-v1:0"
    NOVA_CANVAS = 'amazon.nova-canvas-v1:0'
    NOVA_REAL = "amazon.nova-reel-v1:0"


class BedrockWrapper():
    def __init__(self, region='us-west-2'):
        self.region = region
        self.client = boto3.client(
            service_name="bedrock",
            region_name=self.region,
        )

    '''
    Check to foundation models
    '''
    def list_foundation_models(self):
        return self.client.list_foundation_models()
    
    def decribe_foundation_model(self):
        return self.client.get_foundation_model(
            modelIdentifier=self.modelId
        )
