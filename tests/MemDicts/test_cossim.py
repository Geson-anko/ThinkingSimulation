from MemDicts.cossim import CosSimMemDict
import torch

def test_instance():
    md =  CosSimMemDict(10,3)
    assert md.weight.size(0) == 10
    assert md.weight.size(1) == 3

def test_connect():
    num_memory = 10
    num_dims = 64
    md = CosSimMemDict(num_memory,num_dims,lr=10) 
    src_ids = torch.arange(num_memory)
    tgt_ids = [torch.arange(num_memory) for _ in range(num_memory)]
    md.connect(src_ids,tgt_ids)

    return md, src_ids

def test_trace():
    num_memory = 10
    num_dims = 64
    md = CosSimMemDict(num_memory,num_dims,lr=10) 
    src_ids = torch.arange(num_memory - 4)
    print("Identical mapping",md.trace(src_ids))

def test_trace_from_connected():
    md, src_ids = test_connect()
    print("Stocastic",md.trace(src_ids[:3]))
    md.stocastic=False
    print("Threshold",md.trace(src_ids[:3]))


def test_add_memories():
    
    num_memory = 10
    num_dims = 64
    md = CosSimMemDict(num_memory,num_dims,lr=10) 
    md.add_memories(10)

    assert md.weight.size(0) == 20

def test_get_memory_vector():
    md = CosSimMemDict(10,64)
    v = md.get_memory_vector([1,2,3])
    assert v.size(0) == 3
    assert v.size(1) == 64
    
def test_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_memory = 10
    num_dims = 64
    md = CosSimMemDict(num_memory,num_dims,device=device)
    src_ids = torch.arange(3)
    tgt_ids = [torch.arange(num_memory) for _ in range(3)]
    md.connect(src_ids, tgt_ids)
    md.trace(src_ids)
    

