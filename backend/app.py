import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval()

celeb_embeddings = {}

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        print("No face detected in:", image_path)
        return None
    
    return face.unsqueeze(0) 

def get_embeddings(img):
    with torch.no_grad():
        emb = resnet(img)    
        return emb.squeeze(0)  

# curr_path = os.path.join(os.getcwd(), "Celebrity Faces Dataset")

# for celeb_name in os.listdir(curr_path):
#     celeb_path = os.path.join(curr_path, celeb_name)
    
    
#     embeddings_list = []

#     for img_name in os.listdir(celeb_path):
#         img_path = os.path.join(celeb_path, img_name)
#         face = preprocess(img_path)

#         if face is None:
#             continue
        
#         emb = get_embeddings(face)
#         embeddings_list.append(emb)

#     if len(embeddings_list) == 0:
#         print("No valid embeddings for:", celeb_name)
#         continue

    
#     stacked = torch.stack(embeddings_list)
   
    
#     mean_embedding = stacked.mean(dim=0)
    
#     mean_embedding = mean_embedding / mean_embedding.norm()
    
#     celeb_embeddings[celeb_name] = mean_embedding


# torch.save(celeb_embeddings, "celeb_embeddings.pt")

data = torch.load("celeb_embeddings.pt")
print(data['Brad Pitt'])
