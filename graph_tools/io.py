import networkx as nx
import matplotlib.pyplot as plt
import glob
import os

def save_graph(G:nx.Graph,file_path:str) -> None:
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig(file_path)
    plt.clf()

def get_adjacency_mat_paths(graph_dir:str) -> list[str]:
    """adjacency matrixを保存しているファイルパスを取得します。
    graph_dir/**/adjacency_mat.npy 
    となっていることを想定しています。
    """
    path_param = f"{graph_dir}/**/adjacency_mat.npy"
    return glob.glob(path_param)
