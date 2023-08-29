from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import  RedirectResponse , HTMLResponse 
import uvicorn
import pdfplumber
import nltk
import re
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from minio import Minio
import os
import torch
app = FastAPI()
templates = Jinja2Templates(directory="templates")
model_path = "sup-simcse-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
nltk.download('stopwords')
MONGODB_URL = "mongodb://root:example@mongo:27017/"
client = MongoClient(MONGODB_URL)
db = client["Candidate"]


if "Candidate" not in client.list_database_names():
    client["Candidate"].create_collection("stored_data")
    client["Candidate"].create_collection("job_descriptions")

    
data_collection = db["stored_data"]
job_descriptions_collection = db["job_descriptions"]
MINIO_URL = os.getenv("MINIO_URL", "minio:9000")
minio_client = Minio(
    MINIO_URL,  
    access_key="minio_user",  
    secret_key="minio_password",  
    secure=False
)
bucket_name = "cvs-bucket"  
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)


def get_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    return text

def clean_text(text):
    text = re.sub(r'\(cid:\d+\)', '', text)
    emails = re.findall(r'\S+@\S+', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    phone_number = re.search(r'(\+\d+)', text)
    if phone_number:
        phone_number = phone_number.group(1)
        text = re.sub(r'(\+\d+)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    stop = stopwords.words('english')
    text = ' '.join(word for word in text.split() if word not in stop)
    text = re.sub(r"(?i)\b(\w+)\b", lambda m: m.group(1).lower(), text)
    cleaned_text = " ".join(text.split())
    return cleaned_text, phone_number, emails

class Data(BaseModel):
    cv_pdf: UploadFile = File(..., description="CV file (PDF, TXT)")

@app.get("/docs", response_class=HTMLResponse)
async def custom_swagger_ui_html(request: Request):
    template = get_swagger_ui_template()
    return templates.TemplateResponse(template, {"request": request}.encode('utf-8'))

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

def get_swagger_ui_template():
    return """
    {% extends "swagger-ui-full-layout.html" %}
    {% block head_scripts %}
    {{ super() }}
    <script type="text/javascript">
        window.onload = function() {
            var cvInput = document.getElementById("input_cv_pdf");
            // ... (other inputs)
        };
    </script>
    {% endblock %}
    """


@app.post("/Get_Score")
def Get_Score(    
    full_name=Form(..., description="Full Name"),
    date_of_birth=Form(..., description="Date of Birth (YYYY-MM-DD)"),
    location=Form(..., description="Location"),
    linkedin=Form(None, description="LinkedIn Profile"),
    github=Form(None, description="GitHub Profile"),
    job_position: str=Form(..., description="Desired Job Position !"),
    cv_pdf: UploadFile = File(...),
):
    job_position = job_position.lower()
    cv_pdf_path = cv_pdf.file
    job_description_doc = job_descriptions_collection.find_one({"position": job_position})
    if job_description_doc:
        job_description = job_description_doc["description"]
    else:
        job_description = "No job description available for the selected position."
    
    job_description_text, _, _ = clean_text(job_description)
    job_description_embd = get_embeddings(job_description_text)


    if cv_pdf_path:
        cv_text = extract_text_from_pdf(cv_pdf_path)
        cleaned_text, phone_number, emails = clean_text(cv_text)
        cv_embd = get_embeddings(cleaned_text)
    else:
        print("No file selected for CV.")
 

    # Calculate similarity scores using cosine similarity
    cosine_similarity = 1 - cosine(job_description_embd[0], cv_embd[0])
    score = str(round(float(cosine_similarity) * 100, 2)) + "%"

    # Construct the output JSON data
    output_data = {
        "Full Name": full_name,
        "Date Of Birth": date_of_birth,
        "Location": location,
        "Emails": emails,
        "Phone Number": phone_number,
        "Linkedin": linkedin,
        "Github": github,
        "Desired Job Position": job_position,
        "Score": score
    }

    
    if cv_pdf_path:
        cv_pdf.file.seek(0)
        minio_client.fput_object(bucket_name, cv_pdf.filename, cv_pdf.file.fileno())
        cv_pdf_url = f"cv stored in http://minio:9000/{bucket_name}/{cv_pdf.filename}"
    
    
    data_collection.insert_one({
        "Full Name": full_name,
        "Date Of Birth": date_of_birth,
        "Location": location,
        "Emails": emails,
        "Phone Number": phone_number,
        "Linkedin": linkedin,
        "Github": github,
        "Desired Job Position": job_position,
        "Score": score,
        "cv_pdf_url": cv_pdf_url 
    })


    return output_data


if __name__ == "__main__":
    nltk.download('stopwords')
    uvicorn.run(app, host="127.0.0.1", port=5000)