# pip install fastapi[all] uvicorn tensorflow
# uvicorn test_fastapi:app --host 0.0.0.0 --port 8001 --reload
# nohup uvicorn test_fastapi:app --host 0.0.0.0 --port 8001 --reload > output.log 2>&1 &
# netstat -tuln | grep 8001
# lsof -i :8001
from fastapi.logger import logger as fastapi_logger
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from PIL import Image
import numpy as np
import os, torch, open_clip, logging, pydicom, io, sys


"""
If you are using the images on EDS remote server, e.g.: 
/mnt/eds_share/share/Spine2D/GlobusSrgMapData_crop_square/test/images/anon_2015313.dic_anteroposterior_fullpadding_lumbar_thoracic.jpg
2 ways for plane detection:
(1) In your web browser, enter
    http://10.10.232.240:8001/image_plane?filepath=/mnt/eds_share/share/Spine2D/GlobusSrgMapData_crop_square/test/images/anon_2015313.dic_anteroposterior_fullpadding_lumbar_thoracic.jpg
(2) In your command prompt or terminal, enter
    curl -X POST "http://10.10.232.240:8001/image_plane" -H "Content-Type: application/json" -d "{\"filepath\":\"/mnt/eds_share/share/Spine2D/GlobusSrgMapData_crop_square/test/images/anon_2015313.dic_anteroposterior_fullpadding_lumbar_thoracic.jpg\"}"

If you are uploading image for plane detection, e.g.: C:/Users/yzhou/OneDrive - Globus Medical/Desktop/test.dcm
2 ways for plane detection:
(1) In your web browser, upload your images for plane detection
    http://10.10.232.240:8001/
(2) In your command prompt or terminal, enter
    curl -X POST "http://10.10.232.240:8001/image_plane" -F "localfile=@C:/Users/yzhou/OneDrive - Globus Medical/Desktop/test.dcm"

"""


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
pretrained_model = '/mnt/eds_share/Users/yilu.zhou/Development/log/open_clip_GlobusSrgMapData_crop_square/2023_08_14-13_20_56-model_coca_ViT-L-14-lr_1e-06-b_32-j_4-p_amp/checkpoints/epoch_22.pt'
model, tokenizer, preprocess = load_classify_model(pretrained_model=pretrained_model)

# Adjust path based on whether the application is bundled or run as a script
if getattr(sys, 'frozen', False):
    # The application is bundled (frozen)
    base_dir = sys._MEIPASS
else:
    # The application is run from a script
    base_dir = os.path.dirname(os.path.abspath(__file__))

# Set the paths for static files and templates
static_dir = os.path.join(base_dir, 'static')
#if static_dir doesn't exist, create it
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
template_dir = base_dir

# create FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)


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
    if source in ['curl', 'GET']:
        return {"image_classification": text}
    else:
        return HTMLResponse(f"<h2>image classification: {text}</h2>")


def load_image(filepath, source):
    try:
        dicom_image = pydicom.dcmread(filepath, force=True)
        # Set the TransferSyntaxUID if it's missing
        if not hasattr(dicom_image.file_meta, 'TransferSyntaxUID'):
            dicom_image.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian        
        image_data = dicom_image.pixel_array
        
        if len(image_data.shape) != 2:
            return return_source(source, "ERROR: This DICOM file is not a 2D image")
        
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
            img = return_source(source, f"ERROR: Invalid image format: {e}")
    return img
    

@app.get("/image_plane")
async def image_plane(filepath: str):
    fastapi_logger.info(f"Received GET request for image path: {filepath}")
    source = 'GET'
    # On web browser
    # http://127.0.0.1:8001/image_plane?filepath=/path/to/image.jpg
    # http://10.10.232.240:8001/image_plane?filepath=/mnt/eds_share/share/Spine2D/GlobusSrgMapData_crop_square/test/images/anon_2015313.dic_anteroposterior_fullpadding_lumbar_thoracic.jpg
    # Classify the image_plane
    sentences_plane = ["anteroposterior", "lateral"]
    sentences_anatomy = ['thoracic only', 'lumbar only', 'thoracic and lumbar']
    
    if not filepath:
        return return_source(source, "ERROR: Filepath not provided")
    if not os.path.exists(filepath):
        return return_source(source, "ERROR: File not exist")

    img = load_image(filepath, source)
    if not isinstance(img, Image.Image):
        return img
    
    try:
        image_plane = classify_image_plane(img=img, sentences=sentences_plane)
        image_anatomy = classify_image_plane(img=img, sentences=sentences_anatomy)
        classification = {
            "image_plane": image_plane,
            "image_anatomy": image_anatomy,
        }
        fastapi_logger.info(f"Classification result: {classification}")
        return return_source(source, classification)
    except Exception as e:
        fastapi_logger.error(f"ERROR in classifying image {filepath}: {e}")
        return return_source(source, f"ERROR in classifying image {filepath}: {e}")


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
    fastapi_logger.info(f"User-Agent: {user_agent}")
    if "Mozilla" in user_agent:
        # The request was made by a web browser or another client
        source = f"a client using web browser: {user_agent}"
    else:
        # The request was made by curl or other command line tools
        source = "curl"
    
    # Classify the image_plane
    sentences_plane = ["anteroposterior", "lateral"]
    sentences_anatomy = ['thoracic only', 'lumbar only', 'thoracic and lumbar']
    
    # Check for form data (i.e., uploaded file)
    form_data = await request.form()
    localfile = form_data.get("localfile")
    if localfile:
        fastapi_logger.info(f"Received POST request (made by {source}) for a local file {localfile}")
        # It's a file from client's computer
        file_contents = await localfile.read()
        filepath = io.BytesIO(file_contents)
        
        img = load_image(filepath, source)
        if not isinstance(img, Image.Image):
            return img        
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
        img = load_image(filepath, source)
        if not isinstance(img, Image.Image):
            return img

    try:
        image_plane = classify_image_plane(img=img, sentences=sentences_plane)
        image_anatomy = classify_image_plane(img=img, sentences=sentences_anatomy)
        classification = {
            "image_plane": image_plane,
            "image_anatomy": image_anatomy,
        }
        fastapi_logger.info(f"Classification result: {classification}")
        
        if source != 'curl':
            width, height = img.size
            max_width = 600
            if width > max_width:
                # Calculate the new height to maintain the aspect ratio
                new_height = int((max_width / width) * height)
                # Resize the image
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
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