from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os

celeb_image_urls = {
    "Angelina Jolie": "https://upload.wikimedia.org/wikipedia/commons/a/ad/Angelina_Jolie_2_June_2014_%28cropped%29.jpg",
    "Brad Pitt": "https://upload.wikimedia.org/wikipedia/commons/9/90/Brad_Pitt-69858.jpg",
    "Denzel Washington": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Denzel_Washington_cropped_02_b.jpg",
    "Hugh Jackman": "https://upload.wikimedia.org/wikipedia/commons/6/65/Hugh_Jackman_%288302902601%29.jpg",
    "Jennifer Lawrence": "https://upload.wikimedia.org/wikipedia/commons/5/56/Jennifer_Lawrence_83rd_Oscars.jpg",
    "Johnny Depp": "https://upload.wikimedia.org/wikipedia/commons/7/79/Johnny_Depp_Deauville_2019.jpg",
    "Kate Winslet": "https://upload.wikimedia.org/wikipedia/commons/9/99/Kate_Winslet_at_The_Dressmaker_event_TIFF_%28headshot%29.jpg",
    "Leonardo DiCaprio": "https://upload.wikimedia.org/wikipedia/commons/8/8f/LeonardoDiCaprioNov08.jpg",
    "Megan Fox": "https://upload.wikimedia.org/wikipedia/commons/e/e9/Megan_Fox_2014.jpg",
    "Natalie Portman": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Natalie_Portman_%2848470988352%29_%28cropped%29.jpg",
    "Nicole Kidman": "https://upload.wikimedia.org/wikipedia/commons/2/28/Nicole_Kidman_Cannes_2017_2.jpg",
    "Robert Downey Jr": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Robert_Downey%2C_Jr._2012.jpg",
    "Sandra Bullock": "https://upload.wikimedia.org/wikipedia/commons/1/15/Sandra_Bullock_in_July_2013.jpg",
    "Scarlett Johansson": "https://upload.wikimedia.org/wikipedia/commons/c/c5/Scarlett_Johansson_in_Kuwait_01b-tweaked.jpg",
    "Tom Cruise": "https://upload.wikimedia.org/wikipedia/commons/1/11/Tom_Cruise_May_2022_%28cropped%29.jpg",
    "Tom Hanks": "https://upload.wikimedia.org/wikipedia/commons/9/98/Tom_Hanks_face.jpg",
    "Will Smith": "https://upload.wikimedia.org/wikipedia/commons/8/84/Will-smith-userbox.jpg?_=20111130192147",
}


from app import preprocess_img, get_embeddings, get_min_similarity 

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
            "matched": celeb_name,
            "matched_img_url":celeb_image_urls[celeb_name]
        }

    except Exception as e:
        return {
            "error": "Something went wrong",
            "details": str(e)
        }


