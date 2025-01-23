from pydantic import BaseModel
from typing import List, Optional


class GetPostResponse(BaseModel):
    id: int
    title: str
    content: str
    image_url: str


class UploadPostRequest(BaseModel):
    title: str
    image_url: str
    content: str


class UploadPostResponse(BaseModel):
    id: int
    title: str
    image_url: str
    content: str


class FaceSwapRequest(BaseModel):
    target_image_url: str
    source_image_url: str


class ImageResponse(BaseModel):
    image: bytes
