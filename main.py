from fastapi import FastAPI, UploadFile, File, HTTPException
from schemas import (
    GetPostResponse,
    UploadPostRequest,
    UploadPostResponse,
    FaceSwapRequest,
    ImageResponse,
)
from typing import List
import os
import shutil
import json


app = FastAPI()

POSTS_DIR = "posts"
IMAGES_DIR = "images"
POSTS_FILE = os.path.join(POSTS_DIR, "posts.json")
os.makedirs(POSTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# Helper function to load and save posts
def load_posts():
    with open(POSTS_FILE, "r") as f:
        return json.load(f)


def save_posts(posts):
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=4)


@app.post("/upload-image", response_model=ImageResponse)
def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(IMAGES_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"image_url": f"/images/{file.filename}"}


@app.get("/posts", response_model=List[GetPostResponse])
def get_posts():
    return load_posts()


@app.post("/upload-post", response_model=UploadPostResponse)
def upload_post(request: UploadPostRequest):
    posts = load_posts()
    post_id = len(posts) + 1
    post = {
        "id": post_id,
        "title": request.title,
        "content": request.content,
        "image_url": request.image_url,
    }
    posts.append(post)
    save_posts(posts)
    return post


@app.post("/face-swap")
def face_swap(request: FaceSwapRequest):
    # Placeholder for face swap logic
    target_image_path = os.path.join(
        IMAGES_DIR, os.path.basename(request.target_image_url)
    )
    source_image_path = os.path.join(
        IMAGES_DIR, os.path.basename(request.source_image_url)
    )

    if not os.path.exists(target_image_path) or not os.path.exists(source_image_path):
        raise HTTPException(status_code=404, detail="One or both images not found.")

    swapped_image_path = os.path.join(
        IMAGES_DIR, f"swapped_{os.path.basename(request.target_image_url)}"
    )
    # Simulate face swap by copying the target image as the result
    shutil.copy(target_image_path, swapped_image_path)

    return {"swapped_image_url": f"/images/{os.path.basename(swapped_image_path)}"}


@app.get("/images/{filename}")
def get_image(filename: str):
    file_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return ImageResponse(file_path)
