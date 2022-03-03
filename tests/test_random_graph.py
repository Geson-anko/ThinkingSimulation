from random_graph import generate_directed,generate_undirected,random_generate
import numpy as np

def test_generate_directed():
    adj = generate_directed(4,0.5)
    print("generate_directed",adj)

def test_generate_undirected():
    adj = generate_undirected(6,0.5)
    assert (adj == adj.T).all()
    adj = generate_undirected(5,0.5)
    assert (adj == adj.T).all()
    adj = generate_undirected(4,0.5)
    assert (adj == adj.T).all()
    print("generate_undirected",adj)

def test_random_generate():
    adj = random_generate(4,0.5,True,"all")
    assert np.diag(adj).all()
    adj = random_generate(4,0.5,True,"no")
    assert np.sum(np.diag(adj)) == 0
    adj = random_generate(4,0.5,True,"allow")
    print("random_generate",adj)

    
