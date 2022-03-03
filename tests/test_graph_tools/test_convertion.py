from graph_tools.convertion import adjac2pairs
import numpy as np

def test_adjac2pairs():
    am = np.array([[True,False],[False,True]])
    tgt_pairs = [(0,0),(1,1)]
    pairs = adjac2pairs(am)
    assert pairs == tgt_pairs