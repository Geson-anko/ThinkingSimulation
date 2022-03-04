from graph_tools.io import get_graph_names

def test_get_graph_names():
    graph_paths = ["graphs/abc/mat.npy"]
    out = get_graph_names(graph_paths)
    assert out == ["abc"]