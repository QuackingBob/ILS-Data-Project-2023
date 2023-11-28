import torch
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
import tqdm

cos_sim = CosineSimilarity(dim=1, eps=1E-6)

def dist(v1, v2):
    return cos_sim(v1, v2)

def get_embeddings(nodes, tokenizer, text_encoder, model, device):
    text_inputs = tokenizer(
        nodes, 
        padding="max_length", 
        return_tensors="pt",
        ).to(device)
    # text_features = model.get_text_features(**text_inputs)
    text_embeddings = torch.flatten(text_encoder(text_inputs.input_ids.to(device))['last_hidden_state'],1,-1) # better results when cosine similarity is applied to flattened embeddings

    return text_embeddings

def batch_embeddings(nodes, tokenizer, text_encoder, model, device, batch_size):
    text_inputs = tokenizer(
        nodes, 
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    input_ids = text_inputs.input_ids

    batched_input_ids = torch.split(input_ids, batch_size, dim=0)
    batched_embeddings = []

    for batch_ids in tqdm.tqdm(batched_input_ids):
        batch_embeddings = torch.flatten(text_encoder(batch_ids)['last_hidden_state'], 1, -1)
        batched_embeddings.append(batch_embeddings)

    result_embeddings = torch.cat(batched_embeddings, dim=0)

    return result_embeddings


def get_models(device, index=1):
    models = [
        'openai/clip-vit-base-patch16',
        'openai/clip-vit-base-patch32',
        'openai/clip-vit-large-patch14',
    ]
    
    model_id = models[index % len(models)]
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    text_encoder = CLIPTextModel.from_pretrained(model_id).to(device)
    text_encoder
    model = CLIPModel.from_pretrained(model_id).to(device)

    return tokenizer, text_encoder, model

def get_nodes_from_file(file):
    with open(file, "r", encoding="utf-8") as f:
        nodes = f.readlines()
    return nodes
