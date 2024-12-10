from enum import Enum


class MediaType(Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
  
    @classmethod
    def from_string(cls, string_value):
        for member in cls:
            if member.value == string_value:
                return member
        return None