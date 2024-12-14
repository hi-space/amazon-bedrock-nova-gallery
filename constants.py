from enum import Enum

VIDEO_PREFIX = "video"
IMAGE_PREFIX = "image"
VIDEO_OUTPUT_FILE = "output.mp4"

class MediaType(Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
  
    @classmethod
    def from_string(cls, string_value):
        for member in cls:
            if member.value == string_value:
                return member
        return None
    
class EditingMode(Enum):
    IMAGE_VARIATION = "IMAGE_VARIATION"
    INPAINTING = "INPAINTING"
    OUTPAINTING = "OUTPAINTING"
    IMAGE_CONDITIONING = "IMAGE_CONDITIONING"
    BACKGROUND_REMOVAL = "BACKGROUND_REMOVAL"
