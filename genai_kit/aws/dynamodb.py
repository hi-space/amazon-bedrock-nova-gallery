import boto3
import json
from decimal import Decimal
from datetime import datetime


class DynamoDB:
    def __init__(self, table_name):
        self.db = boto3.resource('dynamodb')
        self.name = table_name
        self.table = self.db.Table(table_name)
        
    def get_item(self, key):
        response = self.table.get_item(Key={
            'id': key
        })
        return json.loads(json.dumps(response.get('Item'), default=_default_serializer))
    
    def put_item(self, item: dict):
        self.table.put_item(
            Item=json.loads(json.dumps(item, default=_default_serializer), parse_float=Decimal)
        )

    def update_item(self, id: str, updates: dict):
        update_expression = "SET " + ", ".join([f"#{k} = :{k}" for k in updates.keys()])
        expression_attribute_names = {f"#{k}": k for k in updates.keys()}
        expression_attribute_values = {f":{k}": json.loads(
            json.dumps(v, default=_default_serializer), 
            parse_float=Decimal
        ) for k, v in updates.items()}
        
        self.table.update_item(
            Key={"id": id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )

    def delete_item(self, id):
        self.table.delete_item(Key={"id": id})
        
    def scan_items(self, query):
        return self.table.scan(**query)


def _default_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
