import torch
from torch.nn import CosineSimilarity
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
import tqdm
import numpy as np
import pandas as pd
import networkx as nx

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
            truncation=True,
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

def graph_to_adjacency_matrix(graph):
    nodes = list(graph.nodes())
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))

    for i, source_node in enumerate(nodes):
        for j, target_node in enumerate(nodes):
            if graph.has_edge(source_node, target_node):
                adjacency_matrix[i, j] = graph[source_node][target_node]['weight']

    return nodes, adjacency_matrix

def construct_graph(nodes:dict, prompts:pd.DataFrame, threshold, tokenizer, text_encoder, model, device):
    # TODO optimize by running graph generation in parallel on multiple threads and then adding up the adjacency matrices
    G = nx.Graph()
    prompt_embeddings = unbatch_embeddings(batch_embeddings(prompts["text"].to_list(), tokenizer, text_encoder, model, device, batch_size=1024))

    for prompt_embedding in tqdm.tqdm(prompt_embeddings, total=prompts.shape[0]):
        prompt_embedding = prompt_embedding.unsqueeze(0).to(device)
        nodes_to_connect = []
        for node_id, node_embedding in nodes.items():
            if dist(prompt_embedding, node_embedding) > threshold:
                nodes_to_connect.append(node_id)

        for i in range(len(nodes_to_connect)):
            for j in range(i + 1, len(nodes_to_connect)):
                node1 = nodes_to_connect[i]
                node2 = nodes_to_connect[j]
                if G.has_edge(node1, node2):
                    G[node1][node2]['weight'] += 1
                else:
                    G.add_edge(node1, node2, weight=1)

    return G

# more optimized than the previous function and directly computes the adjacency matrix (this is a bidirectional graph so I can just to matmul with the transpose)
def construct_graph_adjacency(nodes:dict, prompts:pd.DataFrame, threshold, tokenizer, text_encoder, model, device):
    # TODO optimize by running graph generation in parallel on multiple threads and then adding up the adjacency matrices
    with torch.no_grad():
        adjacency_mat = torch.zeros((len(nodes), len(nodes))).to(device)
        prompt_embeddings = unbatch_embeddings(batch_embeddings(prompts["text"].to_list(), tokenizer, text_encoder, model, device, batch_size=1024))

        for prompt_embedding in tqdm.tqdm(prompt_embeddings, total=prompts.shape[0]):
            prompt_embedding = prompt_embedding.unsqueeze(0).to(device)

            node_embeddings_tensor = torch.stack(list(nodes.values()))
            similarities = (dist(prompt_embedding, node_embeddings_tensor) > threshold).int()
            temp_adj = similarities.T * similarities
            # nodes_to_connect = torch.nonzero(similarities).squeeze().tolist()

            # temp_adj = torch.zeros_like(adjacency_mat)

            # for node in nodes_to_connect:
            #     temp_adj[node] = similarities
            # ^ i was being stupid and then realized that if x is the tensor representing nodes I want to connect bidirectionally, while I could just set those indices in the adj matrix A to x, I could also just take the transpose and do a matrix multiplication of x.T * x
            adjacency_mat += temp_adj
        return adjacency_mat.cpu().numpy()
