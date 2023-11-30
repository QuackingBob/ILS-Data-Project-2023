import os
import json
from utils import *
import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
import tqdm

def main():
    # WARNING: cpu would be very slow
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer, text_encoder, model = get_models(device)

    data_path = r"data/raw/flavors.txt"
    output_dir = r"data/processed"
    file_name = "flavors_combine_redundant.json"

    theta = 0.98
    batch_size = 32

    nodes = get_nodes_from_file(data_path)

    json_dict = {
        "name" : file_name
    }

    node_embeddings = batch_embeddings(nodes, tokenizer, text_encoder, model, device, batch_size)

    filtered_nodes = []

    for name, embed in tqdm.tqdm(zip(nodes, node_embeddings), total=len(nodes)):
        added = False
        for i, node in enumerate(filtered_nodes):
            if dist(embed, node["n1"][1]) >= theta:
                filtered_nodes[i][f"n{len(node) + 1}"] = (name, embed)
                added = True
                break
        if not added:
            filtered_nodes.append({"n1": (name, embed)})
    
    json_dict["nodes"] = filtered_nodes

    with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as outfile:
        json.dump(json_dict, outfile)

if __name__ == "__main__":
    main()