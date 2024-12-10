import boto3
import json
from botocore.config import Config

from langchain_aws.chat_models import ChatBedrock
from langchain.callbacks import StdOutCallbackHandler


class BedrockClaude():
    def __init__(self, region='us-west-2', modelId = 'anthropic.claude-3-5-sonnet-20240620-v1:0', **model_kwargs):
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

        self.model_kwargs = {
            'anthropic_version': 'bedrock-2023-05-31',
            "max_tokens": 4096, # max tokens
            "temperature": 0.1, # [0, 1]
            "top_p": 0.9, # [0, 1]
            "top_k": 250, # [0, 500]
            "stop_sequences": ['Human:', 'H: ']
        }
        self.model_kwargs.update(model_kwargs)

        self.inference_config = {
            'temperature': self.model_kwargs.get('temperature', 0.1),
            'maxTokens': self.model_kwargs.get('max_tokens_to_sample', 4096),
            'topP': self.model_kwargs.get('top_p', 0.9),
            'stopSequences': self.model_kwargs.get('stop_sequences', ['Human:', 'H: '])
        }        
        self.additional_model_fields = {
            'top_k': self.model_kwargs.get('top_k', 200),
        }

    '''
    Langchain API: get ChatBedrock
    '''
    def get_chat_model(self, callback=StdOutCallbackHandler(), streaming=True):
        return ChatBedrock(
            model_id = self.modelId,
            client = self.bedrock,
            streaming = streaming,
            callbacks = [callback],
            model_kwargs = self.model_kwargs,
        )

    '''
    Bedrock API: invoke LLM model
    '''    
    def invoke_llm(self, text: str, image: str = None, system: str = None):
        '''
        Returns:
            dict: ['id', 'type', 'role', 'content', 'model', 'stop_reason', 'stop_sequence', 'usage']
        '''
        parameter = self.model_kwargs.copy()
      
        content = []
        # text
        if text:
            content.append({
                'type': 'text',
                'text': text,
            })
        
        # image
        if image:
            content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/webp',
                    'data': image,
                }
            })

        # system
        if system:
            parameter['system'] = system
        
        parameter.update({
            'messages': [{
                'role': 'user',
                'content': content
            }]
        })

        try:
            response = self.bedrock.invoke_model(
                body=json.dumps(parameter),
                modelId=self.modelId,
                accept='application/json',
                contentType='application/json'
            )
            return json.loads(response.get('body').read())
        except Exception as e:
            print(e)
            return None
        
    def invoke_llm_response(self, text: str, image: str = None, system: str = None):
        return self.invoke_llm(
            text=text, image=image, system=system).get('content', [])[0].get('text', '')

    '''
    Bedrock Converse API
    '''
    def converse(self, text: str, image: bytes = None, system: str = None):
        '''
        Returns:
            str: 어시스턴트의 응답 텍스트
        '''
        # 메시지 내용 구성
        content = []
        if text:
            content.append({'text': text})

        if image:
            content.append({
                'image': {
                    'format': 'png', # png, jpeg, gif, webp
                    'source': {
                        'bytes': image,
                    }
                }
            })

        # 사용자 메시지 생성
        message = {
            'role': 'user',
            'content': content
        }
        messages = [message]

        # 시스템 프롬프트 설정
        system_prompts = []
        if system:
            system_prompts.append({'text': system})

        try:
            response = self.bedrock.converse(
                modelId=self.modelId,
                messages=messages,
                system=system_prompts,
                inferenceConfig=self.inference_config,
                # additionalModelRequestFields=self.additional_model_fields,
            )
            return response
        except Exception as e:
            print(e)
            return None

    '''
    Bedrock Converse Stream
    '''
    def converse_stream(self, text: str, image: bytes = None, system: str = None):
        '''
        Generator that yields assistant's response chunks
        '''
        # 메시지 내용 구성
        content = []
        if text:
            content.append({'text': text})

        if image:
            content.append({
                'image': {
                    'format': 'png', # png, jpeg, gif, webp
                    'source': {
                        'bytes': image,
                    }
                }
            })

        # 사용자 메시지 생성
        message = {
            'role': 'user',
            'content': content
        }
        messages = [message]

        # 시스템 프롬프트 설정
        system_prompts = []
        if system:
            system_prompts.append({'text': system})

        try:
            response = self.bedrock.converse_stream(
                modelId=self.modelId,
                messages=messages,
                system=system_prompts,
                inferenceConfig=self.inference_config,
                # additionalModelRequestFields=self.additional_model_fields,
            )

            stream = response.get('stream')
            if stream:
                for event in stream:
                    if 'contentBlockDelta' in event:
                        delta = event['contentBlockDelta']['delta']
                        if 'text' in delta:
                            yield delta['text']
        except Exception as e:
            print(e)
            return