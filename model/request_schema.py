from pydantic import BaseModel
from typing import List, Optional

class ImageGenerationRequest(BaseModel):
    prompt: str
    input_images_path: str
    num_steps: int = 50
    negative_prompt: Optional[str] = None
    output_directory: str = "output_images"

class ImageGenerationResponse(BaseModel):
    saved_images: List[str]

class ProgramRequest(BaseModel):
    user_query: str