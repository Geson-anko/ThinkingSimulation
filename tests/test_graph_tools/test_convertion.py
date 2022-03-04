from graph_tools.convertion import adjac2pairs, tracedall2adjac
import numpy as np

def test_adjac2pairs():
    am = np.array([[True,False],[False,True]])
    tgt_pairs = [(0,0),(1,1)]
    pairs = adjac2pairs(am)
    assert pairs == tgt_pairs


def test_tracedall2adjac():
    tracedall = [[0,1],[0]]
    expect = np.array([[True,True],[True,False]])
    out = tracedall2adjac(tracedall)
    assert (out == expect).all()
    assert out.shape == expect.shape