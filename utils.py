import torch
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
import tqdm

cos_sim = CosineSimilarity()

def dist(v1, v2):
    return cos_sim(v1, v2)

def get_embeddings(nodes, tokenizer, text_encoder, model, device):
    with torch.no_grad():
        text_inputs = tokenizer(
            nodes, 
            padding="max_length", 
            return_tensors="pt",
            ).to(device)
        # text_features = model.get_text_features(**text_inputs) 
        text_embeddings = torch.flatten(text_encoder(text_inputs.input_ids.to(device))['last_hidden_state'],1,-1) # better results when cosine similarity is applied to flattened embeddings

        return text_embeddings.to("cpu")

def batch_embeddings(nodes, tokenizer, text_encoder, model, device, batch_size, max_batch=1e4):
    with torch.no_grad():
        text_inputs = tokenizer(
            nodes, 
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        print(f"{len(nodes)} nodes")

        input_ids = text_inputs.input_ids

        batched_input_ids = torch.split(input_ids, batch_size, dim=0)
        print(f"{len(batched_input_ids)} batches")
        batched_embeddings = []
        batch_num = 0

        for batch_ids in tqdm.tqdm(batched_input_ids):
            batch_embeddings = torch.flatten(text_encoder(batch_ids)['last_hidden_state'], 1, -1)
            batched_embeddings.append(batch_embeddings.to("cpu"))
            batch_num += 1
            if batch_num > max_batch:
                break

        # result_embeddings = torch.cat(batched_embeddings) # uses too much memory

    # return result_embeddings
    return batched_embeddings


def unbatch_embeddings(batched_embeddings):
    return torch.cat(batched_embeddings)


def get_models(device, index=1):
    models = [
        'openai/clip-vit-base-patch16',
        'openai/clip-vit-base-patch32',
        'openai/clip-vit-large-patch14',
    ]
    
    model_id = models[index % len(models)]
    
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    # text_encoder = CLIPTextModel.from_pretrained(model_id).to(device).half()
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder").half().to(device)
    text_encoder.eval()
    model = CLIPModel.from_pretrained(model_id).to(device)

    return tokenizer, text_encoder, model

def get_nodes_from_file(file):
    with open(file, "r", encoding="utf-8") as f:
        nodes = f.readlines()
    return nodes
