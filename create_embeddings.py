import os
import json
from utils import *
import torch
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
import tqdm
import pickle
import sys

def main(colabfp=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer, text_encoder, model = get_models(device)

    data_path = r"data/raw"
    file_paths = ["artists.txt", "mediums.txt", "movements.txt"]
    output_dir = r"data/processed"
    flavor_file_name = "flavors_combine_redundant.json"
    output_file_name = "embeddings.pd"

    if colabfp:
        colab_dir = r"/content/ILS-Data-Project-2023/"
        data_path = colab_dir + data_path
        output_dir = colab_dir + output_dir

    if not os.path.exists(os.path.join(output_dir, flavor_file_name)):
        print("Could not find flavors combined file, please run combine_nodes.py first")
        return

    batch_size = 1024

    nodes = [get_nodes_from_file(os.path.join(data_path, name)) for name in file_paths]

    node_embeddings = [unbatch_embeddings(batch_embeddings(n, tokenizer, text_encoder, model, device, batch_size)) for n in nodes]

    flavor_file = open(os.path.join(output_dir, flavor_file_name), "r")
    flavor_nodes = [n["n1"] for n in json.load(flavor_file)["nodes"]]
    flavor_embeddings = unbatch_embeddings(batch_embeddings(flavor_nodes, tokenizer, text_encoder, model, device, batch_size))
    flavor_file.close()

    embed_dict = {}

    for i, name in enumerate(file_paths):
        embed_dict[name] = list(zip(nodes[i], node_embeddings[i]))
    
    embed_dict["flavors"] = list(zip(flavor_nodes, flavor_embeddings))

    print("saving... ", end="")

    # outfile = open(os.path.join(output_dir, output_file_name), "wb")
    # pickle.dump(embed_dict, outfile)
    # outfile.close()

    torch.save(embed_dict, os.path.join(output_dir, output_file_name))

    print("done")

if __name__ == "__main__":
    use_colab_fp = False
    if "--colabfp" in sys.argv:
        print("using --colabfp")
        use_colab_fp = True
    main(colabfp=use_colab_fp)