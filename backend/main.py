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
import torch
from ddf import crop_and_encode_image, USER_WATERMARK_IDS, apply_faceswap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

origins = {
    "http://localhost:3000",
    "localhost:3000",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 모든 도메인 허용 (배포 시 특정 도메인으로 제한 권장)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

POSTS_DIR = "posts"
IMAGES_DIR = "images"
ABSOLUTE_PATH = "/home/yoojinoh/Others/deepsafe/backend"
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


def process_url(
    url, base_path="/home/yoojinoh/Others/deepsafe/backend/"
):  # Add(Yoojin)
    url = url.lstrip("/")  # 경로 앞에 '/'가 있으면 제거
    full_image_path = os.path.join(base_path, url)
    print(f"[DEBUG] Final image path: {full_image_path}")
    if not os.path.exists(full_image_path):
        raise FileNotFoundError(f"[ERROR] Image not found at: {full_image_path}")
    return full_image_path


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
    if not posts:
        posts = []
    post_id = len(posts) + 1
    post = {
        "id": post_id,
        "user": request.user,
        "title": request.title,
        "content": request.content,
        "image_url": request.image_url,
    }
    image_url = post["image_url"]

    image_url = process_url(image_url)

    # 해당 image_url encoding 하기. ddf main 참고
    if post["user"] == "byeon" or post["user"] == "cha":
        crop_and_encode_image(
            "byeon_cha", image_url, USER_WATERMARK_IDS[post["user"]], device
        )
    elif post["user"] == "win" or post["user"] == "chu":
        crop_and_encode_image(
            "win_chuu", image_url, USER_WATERMARK_IDS[post["user"]], device
        )
    # post["image_url"] = os.path.join(
    #     os.path.dirname(image_url), "encoded_" + os.path.basename(image_url)
    # )
    encoded_image_path = os.path.join(
        os.path.dirname(image_url), "encoded_" + os.path.basename(image_url)
    )

    # [추가] encoded 이미지를 IMAGES_DIR로 이동
    final_encoded_path = os.path.join(IMAGES_DIR, os.path.basename(encoded_image_path))
    shutil.move(encoded_image_path, final_encoded_path)

    post["image_url"] = f"/images/{os.path.basename(encoded_image_path)}"
    posts.append(post)
    save_posts(posts)
    return post


@app.post("/face-swap")
def face_swap(request: FaceSwapRequest):

    target_image_filename = os.path.basename(request.target_image_url)
    source_image_filename = os.path.basename(request.source_image_url)
    target_image_path = os.path.join(ABSOLUTE_PATH, target_image_filename)
    source_image_path = os.path.join(ABSOLUTE_PATH, source_image_filename)
    print(target_image_path, source_image_path)

    if not os.path.exists(target_image_path) or not os.path.exists(source_image_path):
        raise HTTPException(status_code=404, detail="One or both images not found.")

    # face swap 코드 추가

    swapped_image_path = os.path.join(
        IMAGES_DIR, f"swapped_{os.path.basename(request.target_image_url)}"
    )

    for post in load_posts():
        if post["image_url"] == request.source_image_url:
            source_user = post["user"]
            if source_user == "byeon" or source_user == "cha":
                apply_faceswap(
                    model_type="byeon_cha",
                    swapped_image_path=(swapped_image_path),
                    src_path=post["image_url"],
                    tgt_path=target_image_path,
                    src_user=source_user,
                )
            elif source_user == "win" or source_user == "chu":
                apply_faceswap(
                    model_type="win_chuu",
                    swapped_image_path=(swapped_image_path),
                    src_path=post["image_url"],
                    tgt_path=target_image_path,
                    src_user=source_user,
                )
            break

    return {"swapped_image_url": f"/images/{os.path.basename(swapped_image_path)}"}


@app.get("/images/{filename}")
def get_image(filename: str):
    file_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(file_path)
