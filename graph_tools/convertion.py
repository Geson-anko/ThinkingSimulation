import numpy as np
def adjac2pairs(adajcency_mat:np.ndarray) -> list[tuple[int,int]]:
    """adjacency matrix を各ノードのペアとして返します。"""
    am = adajcency_mat
    num_nodes = len(am)
    pairs = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if am[i][j]]
    return pairs

import networkx as nx
def pairs2graph(pairs:list[tuple[int,int]], directed:bool, num_nodes:int=None):
    """pairをGraphに埋め込み、その結果を返します。"""
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    if num_nodes:
        nodes = [*range(num_nodes)]
        G.add_nodes_from(nodes)

    G.add_edges_from(pairs)
    
    return G

def tracedall2adjac(tracedall:list[list[int]]) -> np.ndarray:
    """記憶辞書のすべての記憶をそれぞれtraceした結果を、adjacency matrixに変換します。"""
    num_nodes = len(tracedall)
    adj = np.zeros((num_nodes,num_nodes),dtype=bool)
    for i in tracedall:
        adj[i] = True
    
    return adj