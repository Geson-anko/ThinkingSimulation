from graph_tools.metrics import accuracy,extra,shortage
import numpy as np
adj_mat = np.array([
        [True,True],
        [True,False]
    ])
    
rec_adj_mat = np.array([
        [False,True],
        [True,True]
    ])

def test_accuracy():
    assert accuracy(adj_mat,rec_adj_mat) == 50

def test_extra():
    assert extra(adj_mat,rec_adj_mat) == 25

def test_shortage():
    assert shortage(adj_mat,rec_adj_mat) == 25

    