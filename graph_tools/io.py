import networkx as nx
import matplotlib.pyplot as plt

def save_graph(G:nx.Graph,file_path:str) -> None:
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig(file_path)
    plt.clf()