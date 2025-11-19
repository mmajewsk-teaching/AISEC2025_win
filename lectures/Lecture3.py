from transformers import AutoTokenizer, AutoModel
import torch
import nltk
from nltk.tokenize import sent_tokenize
import re
import numpy as np

device = "cpu"

def create_fixed_size_chunks(text, chunk_size=1000, overlap=0):
    chunks = []
    start = 0
    text_length = len(text)
    while start<text_length:
        end = min(start+chunk_size, text_length)
        if start>0 and overlap>0:
            start = start - overlap
        chunks.append(text[start:end])
        start = end
    return chunks

def get_embeddings(text, tokenizer, model):
    encoded_inputs = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    token_embeddings = outputs.last_hidden_state
    embeddings = torch.mean(token_embeddings, dim=1)
    return embeddings.cpu().numpy()

cosine_similarity = lambda a, b: np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def retrieve_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    similarities=[]

    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((i, similarity, chunks[i]))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

if __name__ == "__main__":
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)


    script_path = "../shrek.txt"
    with open(script_path, "r", encoding="utf-8") as file:
        script_text = file.read()
    script_text = re.sub(" +", " ", script_text)

    chunks = create_fixed_size_chunks(script_text, overlap=50)
    embeddings = get_embeddings(chunks, tokenizer, model)

    text = "who is fiona"

    query_embedding = get_embeddings([text], tokenizer, model)[0]
    top_chunks = retrieve_chunks(query_embedding, embeddings, chunks, top_k=3 )
    for tp in top_chunks:
        print("---------------------------------------------------")
        print(tp[-1])
