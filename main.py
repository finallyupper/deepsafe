from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from schemas import (
    GetPostResponse,
    UploadPostRequest,
    UploadPostResponse,
    FaceSwapRequest,
    ImageResponse,
    ImageRequest,
)
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
import json
import uuid

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (배포 시 특정 도메인으로 제한 권장)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)


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


@app.post("/upload-image")
async def upload_image(request: Request):
    try:
        # 요청에서 바이너리 데이터 읽기
        image_data = await request.body()

        # 파일 저장
        random_filename = (
            f"{uuid.uuid4()}.png"  # Using UUID to generate a random filename
        )
        file_path = os.path.join(IMAGES_DIR, random_filename)

        # 파일 저장
        with open(file_path, "wb") as f:
            f.write(image_data)

        # 이미지 URL 반환
        return {"image_url": f"/images/{random_filename}"}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to process image: {str(e)}"
        )


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
    return FileResponse(file_path)
