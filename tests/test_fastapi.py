# pip install fastapi[all] uvicorn tensorflow
# uvicorn test_fastapi:app --host 0.0.0.0 --port 8001 --reload
# netstat -tuln | grep 8001
# lsof -i :8001
from fastapi.logger import logger as fastapi_logger
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from PIL import Image
import numpy as np
import os, torch, open_clip, logging, pydicom, io

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

# set up log settings
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, filename="fastapi.log", format=log_format)
fastapi_logger.addHandler(logging.StreamHandler())

# load model
pretrained_model = '/mnt/eds_share/Users/yilu.zhou/Development/log/open_clip_GlobusSrgMapData_crop_square/2023_08_28-10_39_59-model_coca_ViT-L-14-lr_5e-06-b_32-j_4-p_amp/checkpoints/epoch_24.pt'
model, tokenizer, preprocess = load_classify_model(pretrained_model=pretrained_model)

# create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=".")


def classify_image_plane (img, sentences):
    """
    Function for image plan classification.
    """
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


def return_source(source, text):
    if source == 'curl':
        return {"image_plane": text}
    else:
        return HTMLResponse(f"<h2>image_plane: {text}</h2>")
    

@app.get("/image_plane")
async def image_plane(filepath: str):
    fastapi_logger.info(f"Received GET request for image path: {filepath}")
    # On web browser
    # http://127.0.0.1:8001/image_plane?filepath=/path/to/image.jpg
    # http://10.10.232.240:8001/image_plane?filepath=/mnt/eds_share/share/Spine2D/GlobusSrgMapData_crop_square/test/images/anon_2015313.dic_anteroposterior_fullpadding_lumbar_thoracic.jpg
    # Classify the image_plane
    sentences = ["anteroposterior", "lateral"]
    
    if not filepath:
        return {"image_plane": "ERROR: Filepath not provided"}
    if not os.path.exists(filepath):
        return {"image_plane": "ERROR: File not exist"}

    try:
        dicom_image = pydicom.dcmread(filepath)
        image_data = dicom_image.pixel_array
        
        # Convert image data to 8-bit pixels
        image_data = (np.maximum(image_data, 0) / image_data.max()) * 255.0
        image_data = np.uint8(image_data)
        
        img = Image.fromarray(image_data)
        if img.mode not in ['L', 'RGB']:
            img = img.convert('L').convert('RGB')
    except:
        try:
            img = Image.open(filepath).convert('RGB')
        except Exception as e:
            return {"image_plane": f"ERROR: Invalid image format: {e}"}
    
    try:
        classification = classify_image_plane(img=img, sentences=sentences)
        fastapi_logger.info(f"Classification result: {classification}")
        return {"image_plane": classification}
    except Exception as e:
        fastapi_logger.error(f"ERROR in classifying image {filepath}: {e}")
        return {"image_plane": f"ERROR in classifying image {filepath}: {e}"}


@app.post("/image_plane")
async def image_plane(request: Request):
    # On command line
    # curl -X POST "http://127.0.0.1:8001/image_plane" -H "Content-Type: application/json" -d "{\"filepath\":\"/path/to/image.jpg\"}"
    """
    curl -X POST "http://10.10.232.240:8001/image_plane" -H "Content-Type: application/json" -d "{\"filepath\":\"/mnt/eds_share/share/Totalsegmentator_dataset/segmentation_for_sahana/DDR/projection/20230906/Mir_Pre-OP_CT_2022-07-22T124202_1.1_0_2_anon_Unknown_Lat_projection_enhance1.dcm\"}"
    
    curl -X POST "http://10.10.232.240:8001/image_plane" -F "localfile=@C:/Users/yzhou/OneDrive - Globus Medical/Desktop/test.dcm"
    """
    # Extract the User-Agent and determine the source of the request
    user_agent = request.headers.get("User-Agent", "")
    
    if "curl" in user_agent:
        # The request was made by curl
        source = "curl"
    else:
        # The request was made by a web browser or another client
        source = f"a client using web browser: {user_agent}"
    
    # Classify the image_plane
    sentences = ["anteroposterior", "lateral"]
    
    # Check for form data (i.e., uploaded file)
    form_data = await request.form()
    localfile = form_data.get("localfile")
    if localfile:
        fastapi_logger.info(f"Received POST request (made by {source}) for a local file {localfile}")
        # It's a file from client's computer
        file_contents = await localfile.read()
        filepath = io.BytesIO(file_contents)
        try:
            dicom_image = pydicom.dcmread(filepath)
            image_data = dicom_image.pixel_array
            
            # Convert image data to 8-bit pixels
            image_data = (np.maximum(image_data, 0) / image_data.max()) * 255.0
            image_data = np.uint8(image_data)
            
            img = Image.fromarray(image_data)            
            if img.mode not in ['L', 'RGB']:
                img = img.convert('L').convert('RGB')
        except:
            # If it's not a DICOM file, try reading it as a regular image
            try:
                with filepath as f:
                    img = Image.open(f).convert('RGB')
            except Exception as e:
                return return_source(source, f"ERROR: Invalid image format: {e}") #{"image_plane": f"Invalid image format: {e}"}
    else:
        try:
            json_data = await request.json()
            filepath = json_data.get("filepath")
        except Exception as e:
            return return_source(source, f"ERROR: Invalid JSON format: {e}")
        
        if not filepath:
            return return_source(source, "ERROR: Filepath not provided")
        if not os.path.exists(filepath):
            return return_source(source, "ERROR: File not exist")

        fastapi_logger.info(f"Received POST request for image path: {filepath}")
        try:
            # Load the DICOM file using pydicom
            dicom_image = pydicom.dcmread(filepath)
            img = Image.fromarray(dicom_image.pixel_array)
            if img.mode not in ['L', 'RGB']:
                img = img.convert('L').convert('RGB')
        except:
            try:
                img = Image.open(filepath).convert('RGB')
            except Exception as e:
                return return_source(source, f"ERROR: Invalid image format: {e}")

    try:
        classification = classify_image_plane(img=img, sentences=sentences)
        fastapi_logger.info(f"Classification result: {classification}")
        
        if source != 'curl':
            width, height = img.size
            max_width = 600
            if width > max_width:
                # Calculate the new height to maintain the aspect ratio
                new_height = int((max_width / width) * height)
                # Resize the image
                img = img.resize((max_width, new_height), Image.ANTIALIAS)
            temp_image_path = f"static/test.png"
            img.save(temp_image_path)
            # Render the HTML template with results and image
            return templates.TemplateResponse("image_plane.html", {
                "request": request,
                "classification": classification,
                "image_path": temp_image_path,
            })
        else:
            return return_source(source, classification)
    except Exception as e:
        fastapi_logger.error(f"ERROR in classifying image {filepath}: {e}")
        return return_source(source, f"ERROR in classifying image {filepath}: {e}")


@app.get("/")
async def read_template(request: Request):
    return templates.TemplateResponse("image_plane.html", {"request": request, "classification": None, "image_path": None})