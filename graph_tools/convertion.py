import numpy as np
def adjac2pairs(adajcency_mat:np.ndarray) -> list[tuple[int,int]]:
    """adjacency matrix を各ノードのペアとして返します。"""
    am = adajcency_mat
    num_nodes = len(am)
    pairs = [(i,j) for i in range(num_nodes) for j in range(num_nodes) if am[i][j]]
    return pairs

def pairs2graph(pairs:list[tuple[int,int]], directed:bool, num_nodes:int=None):
    pass