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

    theta = 0.95
    batch_size = 512
    max_batch = 4 # have to limit due to computational complexity (you can change and set it up to 198 since there is about 198 batches)

    nodes = get_nodes_from_file(data_path)

    json_dict = {
        "name" : file_name
    }

    node_embeddings = batch_embeddings(nodes, tokenizer, text_encoder, model, device, batch_size, max_batch)

    filtered_nodes = []
    filtered_nodes_names = []

    # commented because the torch.cat operation takes far too much memory and this only works if the batches have been combined
    # for name, embed in tqdm.tqdm(zip(nodes, node_embeddings), total=len(nodes)):
    #     added = False
    #     for i, node in enumerate(filtered_nodes):
    #         if dist(embed.unsqueeze(0).to(device), node["n1"][1].unsqueeze(0).to(device)) >= theta:
    #             filtered_nodes[i][f"n{len(node) + 1}"] = (name, embed.to("cpu"))
    #             added = True
    #             break
    #     if not added:
    #         filtered_nodes.append({"n1": (name, embed)})

    with torch.no_grad():
        for index, name in tqdm.tqdm(enumerate(nodes), total=len(nodes)):
            batch_num = index // batch_size
            sample = index % batch_size
            embed = node_embeddings[batch_num][sample].to(device)
            added = False
            for i, node in enumerate(filtered_nodes):
                if dist(embed.unsqueeze(0), torch.Tensor(node["n1"][1]).unsqueeze(0).to(device)) >= theta: # TODO this operation can be done beforehand
                    node_num = len(node) + 1
                    filtered_nodes[i][f"n{node_num}"] = (name, embed.to("cpu"))
                    filtered_nodes_names[i][f"n{node_num}"] = name
                    added = True
                    break
            if not added:
                filtered_nodes.append({"n1": (name, embed.to("cpu"))})
                filtered_nodes_names.append({"n1": name})
            if batch_num >= max_batch:
                break
    
    json_dict["nodes"] = filtered_nodes_names

    with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as outfile:
        json.dump(json_dict, outfile)

    # I'm not saving the embeddings to a file because they are too large and can be recomputed from the names

if __name__ == "__main__":
    main()