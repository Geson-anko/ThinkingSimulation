from utils import load_setting
from attrdict import AttrDict

def test_load_setting():
    s = load_setting("settings/random_graph_example.json")
    assert type(s) is AttrDict
    print(s)
    