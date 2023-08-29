from transformers import AutoTokenizer,AutoModel
import torch
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import re
import pdfplumber
import nltk
import json
import tkinter as tk
from tkinter import filedialog
nltk.download('stopwords')
model_path ="sup-simcse-roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
def get_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text
root = tk.Tk()
root.withdraw()
job_description_file_path = filedialog.askopenfilename(filetypes=[('PDF Files', '*.pdf')])

root = tk.Tk()
root.withdraw()
cv_pdf_path = filedialog.askopenfilename(filetypes=[('PDF Files', '*.pdf')])
def clean_text(text):
    # Extract email addresses
    text = re.sub(r'\(cid:\d+\)', '', text)
    emails = re.findall(r'\S+@\S+', text)
    if emails :
      text = re.sub(r'\S*@\S*\s?', '', text)

    # Extract phone number
    phone_number = re.search(r'(\+\d+)', text)
    if phone_number:
        phone_number = phone_number.group(1)
        text = re.sub(r'(\+\d+)', '', text)
        # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove hashtags
    text = re.sub(r'#', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs (strings starting with www or http)
    text = re.sub(r'http\S+|www.\S+', '', text)
     # Remove  strings starting with cid + number

    stop = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop])
    text = re.sub(r"(?i)\b(\w+)\b", lambda m: m.group(1).lower(), text)

     # Replace multiple spaces with single space
    cleaned_text  = " ".join(text.split())

    return cleaned_text,phone_number,emails
if job_description_file_path:

    job_description_text = extract_text_from_pdf(job_description_file_path)
    job_description_text ,_,_=clean_text(job_description_text)
    job_description_embd = get_embeddings(job_description_text)
else:
    print("No file selected.")
if cv_pdf_path :
    cv_text = extract_text_from_pdf(cv_pdf_path)
    cleaned_text,phone_number,emails=clean_text(cv_text)
    cv_embd = get_embeddings(cleaned_text)
else :
    print("No file selected.")    




# Calculate similarity scores using cosine similarity
cosine_similarity = 1 - cosine(job_description_embd[0], cv_embd[0])

print(f"The CV has a match score of {cosine_similarity*100:.2f}% with the job description ")




score = float(cosine_similarity)
phone_number = str(phone_number)
output_data = {
    "score": score,
    "emails": emails,
    "phone_number": phone_number
}

# Convert the dictionary to a JSON-formatted string
json_string = json.dumps(output_data, indent=4)

# Write the JSON string to a file
output_file = "output.json"
with open(output_file, "w") as f:
    f.write(json_string)