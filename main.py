from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import io
import faiss
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "tiiuae/falcon-7b-instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

dim = 768
index = faiss.IndexFlatL2(dim)
memory_texts = []

def embed_text(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(inputs["input_ids"]).mean(dim=1)
    return embeddings.numpy()

class ChatInput(BaseModel):
    user_input: str

@app.post("/chat")
def chat_response(input_data: ChatInput):
    inputs = processor(text=input_data.user_input, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=200)
    response = processor.batch_decode(output, skip_special_tokens=True)[0]
    return JSONResponse(content={"response": response})

@app.post("/multi-chat")
async def multimodal_chat(user_text: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    inputs = processor(text=user_text, images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=200)
    response = processor.batch_decode(output, skip_special_tokens=True)[0]
    return {"response": response}

@app.post("/memory-chat")
def chat_with_memory(input_data: ChatInput):
    query = input_data.user_input
    query_vec = embed_text(query)

    if index.ntotal > 0:
        _, I = index.search(query_vec, k=1)
        memory = memory_texts[I[0][0]]
        full_prompt = f"Previously: {memory}\nNow: {query}"
    else:
        full_prompt = query

    response = model.generate(**processor(text=full_prompt, return_tensors="pt"), max_new_tokens=200)
    output_text = processor.batch_decode(response, skip_special_tokens=True)[0]

    memory_texts.append(query)
    index.add(query_vec)

    return {"response": output_text}
