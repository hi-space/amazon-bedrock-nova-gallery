from datetime import datetime
from urllib.parse import urlparse


def format_datetime(time_str: str, seconds=False) -> str:
    dt = datetime.fromisoformat(time_str)
    if seconds:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return dt.strftime("%Y년 %m월 %d일 %H시 %M분")


def extract_key_from_uri(s3_uri):
    parsed_uri = urlparse(s3_uri)
    return parsed_uri.path.split('/')[-1]