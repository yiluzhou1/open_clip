# pip install fastapi[all] uvicorn tensorflow
# uvicorn test_fastapi:app --host 0.0.0.0 --port 8000 --reload
# netstat -tuln | grep 8000
# lsof -i :8000


from fastapi import FastAPI, HTTPException
from PIL import Image
import json, os, torch, open_clip

# Load deep learning model
def load_classify_model(pretrained_model):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained=pretrained_model,
        image_mean = (0.45, 0.45, 0.45),
        image_std = (0.23, 0.23, 0.23),
    )
    tokenizer = open_clip.get_tokenizer('coca_ViT-L-14')
    return model, tokenizer, preprocess

pretrained_model = '/mnt/eds_share/Users/yilu.zhou/Development/log/open_clip_GlobusSrgMapData_crop_square/2023_08_28-10_39_59-model_coca_ViT-L-14-lr_5e-06-b_32-j_4-p_amp/checkpoints/epoch_24.pt'
model, tokenizer, preprocess = load_classify_model(pretrained_model=pretrained_model)

app = FastAPI()

def classify_image_plane (image_path, sentences):
    """
    Function for image plan classification.
    """
    if not image_path:
        return "ERROR: Filepath not provided"
    #file not exist
    if not os.path.exists(image_path):
        return "ERROR: File not exist"

    # load an image
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        return None #skip if image is not readable
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    image = preprocess(img).unsqueeze(0)
    text = tokenizer(sentences)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)    
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        text_probs = text_probs.cpu().tolist()[0]
    # Construct the dictionary
    class_predict_dict = dict(zip(sentences, text_probs))
    # Extract the key with the largest value
    class_predict = max(class_predict_dict, key=class_predict_dict.get)
    return class_predict


@app.get("/image_plane")
async def image_plane(filepath: str):
    # On web browser
    # http://127.0.0.1:8001/image_plane?filepath=/path/to/image.jpg
    # http://10.10.232.240:8001/image_plane?filepath=/mnt/eds_share/share/Spine2D/GlobusSrgMapData_crop_square/test/images/anon_2015313.dic_anteroposterior_fullpadding_lumbar_thoracic.jpg
    # Classify the image_plane
    sentences = ["anteroposterior", "lateral"]
    classification = classify_image_plane(filepath, sentences)
    return {"image_plane": classification}


@app.post("/image_plane")
async def image_plane(request: dict):
    # On command line
    # curl -X POST "http://127.0.0.1:8001/image_plane" -H "Content-Type: application/json" -d "{\"filepath\":\"/path/to/image.jpg\"}"
    # curl -X POST "http://10.10.232.240:8001/image_plane" -H "Content-Type: application/json" -d "{\"filepath\":\"/mnt/eds_share/share/Spine2D/GlobusSrgMapData_crop_square/test/images/anon_2015313.dic_anteroposterior_fullpadding_lumbar_thoracic.jpg\"}"
    filepath = request.get("filepath")
    if not filepath:
        raise HTTPException(status_code=400, detail="Filepath not provided")

    # Classify the image_plane
    sentences = ["anteroposterior", "lateral"]
    classification = classify_image_plane(filepath, sentences)

    # Return the classification
    return {"image_plane": classification}

