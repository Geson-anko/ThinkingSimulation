import numpy as np

def accuracy(adj_mat:np.ndarray, rec_adj_mat:np.ndarray) -> float:
    """ adjacency matrix と再構成されたそれとの正答率を計算します。"""
    assert adj_mat.shape == rec_adj_mat.shape
    num_elements = adj_mat.size
    acc = np.sum(adj_mat == rec_adj_mat) / num_elements * 100
    return acc

def extra(adj_mat:np.ndarray, rec_adj_mat:np.ndarray) -> float:
    """\
    adjacency matrix と再構成されたそれとを比較し、
    余分に生まれてしまった数の割合を返します。
    """
    assert adj_mat.shape == rec_adj_mat.shape
    num_elements = adj_mat.size
    adj_mat = adj_mat.reshape(-1)
    rec_adj_mat = rec_adj_mat.reshape(-1)
    T = adj_mat[rec_adj_mat]
    s =  (T.size - np.sum(T)) / num_elements
    ext = s * 100
    return ext

def shortage(adj_mat:np.ndarray, rec_adj_mat:np.ndarray) -> float:
    """\
    adjacency matrix と再構成されたそれとを比較し、
    足りなかった結合の割合を返します。
    """
    return extra(rec_adj_mat, adj_mat)


