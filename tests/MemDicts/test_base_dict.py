from MemDicts.base_dict import _memory_dictionary
import torch
from typing import *

class Dummy(_memory_dictionary):
    def __init__(self, num_memory: int, num_dims: int, *, device: torch.device = None, dtype: torch.float = None) -> None:
        super().__init__(num_memory, num_dims, device=device, dtype=dtype)    
        
    
    def connect(self, src_ids: torch.Tensor, tgt_ids: Union[List[torch.Tensor], torch.Tensor]) -> None:
        return super().connect(src_ids, tgt_ids)

    def trace(self, src_ids: torch.Tensor) -> torch.Tensor:
        return super().trace(src_ids)
    
    def add_memories(self, num: int) -> None:
        return super().add_memories(num)

    def get_memory_vector(self, src_ids: torch.Tensor) -> torch.Tensor:
        return super().get_memory_vector(src_ids)

def test_format_tgt_ids():
    num_memory = 5
    md = Dummy(num_memory,5)
    tgt_ids = [
                torch.tensor([1,0,3,4,2]),
                torch.tensor([3,2,0]),
            ]

    result = torch.tensor([ # 上記と同じ
                [True,True,True,True,True],
                [True,False,True,True,False]
            ],dtype=torch.bool)
    output = md.format_tgt_ids(tgt_ids)
    assert (output == result).any()
