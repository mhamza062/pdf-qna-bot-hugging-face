from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import PyPDF2

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_falcon_pipeline():  # You can rename this later
    model_name = "tiiuae/falcon-rw-1b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe
