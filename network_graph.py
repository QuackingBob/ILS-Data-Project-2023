import plotly.graph_objects as go
import networkx  as nx
import numpy as np
import os
import json
import tqdm
from random import shuffle
import sys

def get_color(index:int, category_dict:dict, colors:list):
    for i, key in enumerate(list(category_dict.keys())):
        if index in category_dict[key]:
            return colors[i]


def get_top_n_indices(array, n, start_index, end_index):
    sorted_indices = np.argsort(array)[::-1]
    sorted_indices = sorted_indices[(sorted_indices >= start_index) & (sorted_indices <= end_index)]
    top_n_indices = sorted_indices[:n]
    return top_n_indices

def main(colabfp=False):
    dataset_names = ["artists.txt", "mediums.txt", "movements.txt", "flavors"]
    output_dir = r"data/processed"
    output_file_name = "graph.csv"
    output_id_name = "node_ids.json"

    if colabfp:
        colab_dir = r"/content/ILS-Data-Project-2023/"
        output_dir = colab_dir + output_dir

    adjacency_matrix = np.loadtxt(os.path.join(output_dir, output_file_name), delimiter=",")
    json_dict = json.load(open(os.path.join(output_dir, output_id_name)))
    node_ids = json_dict["ids"]
    category_dict = json_dict["categories"]
    colors = ["#011627", "#f71735", "#41ead4", "#fdfffc"] # ["#D6F9DD", "#99F7AB", "#ABDF75", "#60695C"]

    max_edges = 10000
    min_degree = 10
    num_nodes = 64 # 152
    min_weight = 10 # 4000

    print("Loading Graph ... ")
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph())

    node_info = {}
    index_ranges = { cat:(category_dict[cat][0], category_dict[cat][-1]) for cat in list(category_dict.keys())}
    categories = list(category_dict.keys())
    num_comp = 3
    for i in range(adjacency_matrix.shape[0]):
        datastr = ""
        for cat in categories:
            datastr += f"<br>{cat}: "
            n_num = 1
            for idx in get_top_n_indices(adjacency_matrix[i], num_comp, index_ranges[cat][0], index_ranges[cat][1]):
                datastr += f" {n_num}. {node_ids[str(idx)]}"
                if n_num < num_comp:
                  datastr += ","
                n_num += 1
            datastr += "</br>"
        node_info[i] = node_ids[str(i)] + ": " + datastr

    # print stats
    print("\tgraph stats:")
    print(f"\t-nodes: {len(G.nodes)}")
    degrees = [degree for node, degree in G.degree()]
    print(f"\t-min node degree: {min(degrees)}") # -> 1715 we need to reduce nodes in the graph to only the largest or else it is extremely difficult to read
    print(f"\t-max node degree: {max(degrees)}") # -> 7498 
    weights = [np.sum(adjacency_matrix[node]) for node in G.nodes]
    print(f"\t-min node weight: {min(weights)}") # -> 1715 we need to reduce nodes in the graph to only the largest or else it is extremely difficult to read
    print(f"\t-max node weight: {max(weights)}") # -> 7498 

    # filter by degree
    print("\treducing degree")
    remove = [node for node, degree in G.degree() if degree < min_degree]
    G.remove_nodes_from(remove)
    # G.remove_nodes_from(list(nx.isolates(G)))

    # filter by total weight
    print("\tfiltering by weight")
    remove = [node for node in G.nodes if np.sum(adjacency_matrix[node]) < min_weight]
    G.remove_nodes_from(remove)

    # create sample
    print(f"\tsampling {num_nodes} nodes")
    sample = [node for node in G.nodes]
    for category in list(category_dict.keys()):
        sample_in_category = [node for node in sample if node in category_dict[category]]
        shuffle(sample_in_category)
        sample_in_category = sample_in_category[num_nodes // 4:]
        G.remove_nodes_from(sample_in_category)

    print("Done")

    # Extract edge weights from the adjacency matrix
    edge_weights = adjacency_matrix # adjacency_matrix[adjacency_matrix.nonzero()]

    # Extract node colors and labels from the dictionaries
    node_colors = {i: get_color(i, category_dict, colors) for i in G.nodes}
    node_labels = {i: node_ids[str(i)] for i in G.nodes}

    # Create a plotly graph object
    fig = go.Figure(
        layout=go.Layout(
            title='<br>Network graph made with Python',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text=f"Network Graph Showing Top {4 * (num_nodes // 4)} Nodes ",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )

    print("Adding nodes to graph ...")
    for node in tqdm.tqdm(G.nodes):
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers+text',
            text=f"{node_labels[node]}",
            textposition="bottom center",
            hovertemplate=f"{node_info[node]}",
            marker=dict(
                size=10,
                color=node_colors[node],
                line=dict(color='black', width=1)
            ),
            name=str(node)
        ))

    print("Adding edges ... ")
    index = 0
    edges_added = []
    edges = [edge for edge in G.edges]
    shuffle(edges)
    for edge in tqdm.tqdm(edges, total=min(max_edges, len(edges))):
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            line=dict(
                width=np.log(edge_weights[edge[0], edge[1]]),
                color='#AAAAAA',
            ),
            hoverinfo='none',
            showlegend=False,
            opacity=0.5
        ))
        edges_added.append(edge)
        index += 1
        if index >= max_edges:
            break

    print("Updating Layout ... ")
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
    )

    print("Adding node positions ... ")
    # pos = nx.spring_layout(G) # looks cool but really expensive when you have 7497+ nodes :<
    pos = nx.circular_layout(G)
    for i, node in tqdm.tqdm(enumerate(G.nodes)):
        fig.data[i].x = [pos[node][0]]
        fig.data[i].y = [pos[node][1]]

    print("Updating line traces ... ")
    index = 0
    for i, edge in tqdm.tqdm(enumerate(edges_added), total=min(max_edges, len(G.edges))):
        fig.data[len(G.nodes) + i].x = [pos[edge[0]][0], pos[edge[1]][0]]
        fig.data[len(G.nodes) + i].y = [pos[edge[0]][1], pos[edge[1]][1]]
        index += 1
        if index >= max_edges:
            break

    # Show the plot
    fig.show()


if __name__ == "__main__":
    use_colab_fp = False
    if "--colabfp" in sys.argv:
        print("using --colabfp")
        use_colab_fp = True
    main(colabfp=use_colab_fp)
