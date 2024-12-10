import re
import numpy as np
from typing import List, Union


def deep_clean(data):
    if isinstance(data, dict):
        cleaned_data = {k: deep_clean(v) for k, v in data.items() if v is not None and v != ''}
        return {k: v for k, v in cleaned_data.items() if v}
    elif isinstance(data, list):
        cleaned_list = [deep_clean(item) for item in data if item is not None and item != '']
        if cleaned_list:
            return cleaned_list
        else:
            return None
    else:
        return data


def safe_float_conversion(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def softmax(xlist: list):
    x = np.array(xlist)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def extract_xml_values(input_string: str, extract_keyword: str) -> Union[List[str], str]:
    """
    XML 태그 사이의 값을 추출하는 함수
    
    Args:
        input_string (str): XML 태그가 포함된 입력 문자열
        extract_keyword (str): 추출하고자 하는 태그 키워드
        
    Returns:
        Union[List[str], str]: 추출된 값들의 리스트 또는 단일 문자열
        태그가 없는 경우 빈 리스트 반환
    """
    
    pattern = f"<{extract_keyword}>(.*?)</{extract_keyword}>"
    matches = re.findall(pattern, input_string)
    return matches