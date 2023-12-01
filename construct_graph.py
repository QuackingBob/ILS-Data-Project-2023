import os
import json
from utils import *
import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
import tqdm
import networkx as nx
import pandas as pd
import numpy as np

def main():
    # WARNING: cpu would be very slow
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer, text_encoder, model = get_models(device)

    data_path = r"data/raw"
    data_file = "embeddings.pd"
    dataset_names = ["artists.txt", "mediums.txt", "movements.txt", "flavors"]
    output_dir = r"data/processed"
    output_file_name = "graph3.csv"
    output_id_name = "node_ids.json"
    prompt_data = r"prompt data/data/train-00000-of-00001.parquet"

    if not os.path.exists(os.path.join(output_dir, data_file)):
        print("Could not find embeddings file, please run create_embeddings.py first")
        return
    
    embedding_file = open(os.path.join(output_dir, data_file), "rb")
    node_dict = torch.load(embedding_file)

    threshold = 0.85
    num_samples = 4000

    nodes = {}
    node_ids = {}
    ids_categories = {}
    
    curr_id = 0

    for file_cont in dataset_names:
        ids_categories[file_cont] = []
        for i, n in enumerate(node_dict[file_cont]):
            nodes[curr_id] = n[1].unsqueeze(0).to(device)
            node_ids[curr_id] = n[0]
            ids_categories[file_cont].append(curr_id)
            curr_id += 1
    
    if not os.path.exists(os.path.join(output_dir, output_id_name)):
        json_dict = {
            "name" : output_id_name
        }
        json_dict["ids"] = node_ids
        json_dict["categories"] = ids_categories
        with open(os.path.join(output_dir, output_id_name), "w", encoding="utf-8") as outfile:
            json.dump(json_dict, outfile)
    
    data_frame = pd.read_parquet(os.path.join(data_path, prompt_data))
    print(data_frame.info())
    prompts = data_frame.sample(num_samples)

    # graph = construct_graph(nodes, prompts, threshold, tokenizer, text_encoder, model, device)
    # node_list, adjacency_mat = graph_to_adjacency_matrix(graph)
    # much faster by utilizing tensor operations and directly computing the adjacency matrix:
    adjacency_mat = construct_graph_adjacency(nodes, prompts, threshold, tokenizer, text_encoder, model, device)

    # torch.save(adjacency_mat, open(os.path.join(output_dir, output_file_name), "wb"))
    np.savetxt(os.path.join(output_dir, output_file_name), adjacency_mat, delimiter=',')

if __name__ == "__main__":
    main()