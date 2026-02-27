from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app import preprocess_img, get_embeddings, get_min_similarity # use new match function

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def demo():
    return {"message": "Welcome"}

@app.post("/findmatch")
async def findmatch(photo: UploadFile = File(...)):
    try:
        
        image_bytes = await photo.read()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = preprocess_img(img)

        
        emb = get_embeddings(img)

        celeb_name = get_min_similarity(emb)

        return {
            "matched": celeb_name
        }

    except Exception as e:
        return {
            "error": "Something went wrong",
            "details": str(e)
        }