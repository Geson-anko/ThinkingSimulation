"""
一時記憶の実装クラスです。
定義
    一時的に記憶を要素としてそれを保持しておく要素数 𝛼 の有限集合。𝑊の要素数が 𝛼 の時に要素を追加する場
    合、𝑊 の中の要素のどれかがランダムに選ばれ上書きされる。要素数𝑘 | 𝑘 ≤ 𝛼の𝑊と記憶を要素とする要素数
    𝑙 | 𝑙 ≤ 𝛼 個の集合と結合する場合、𝑊 の中から 𝑘 + 𝑙 − 𝛼 個の要素がランダムに選ばれ上書きされる。

The Implementation class of "Working Memory".
*Definition
    Working Memory is a finite set with a maximum number of elements α that holds memories temporarily as an
    element. If the elements are added when the number of elements in W is α, the elements in W are randomly chosen
    and overwritten
"""
import torch
import numpy as np
from typing import *
from logger import *

setLogger(__name__,DEBUG)

class WorkingMemory:
    """
    このクラスは次のように、記憶を格納しておくためのクラスです。
        W = {M_1, M_2, M_3}
    """

    def __init__(self, max_length:int, id_type:np.dtype=np.int64) -> None:
        """
        空のWorking Memory を作成します
        Create a empty Working Memory.
        """

        self.id_type = id_type
        type_info = np.iinfo(id_type)
        self.min_id = 0
        self.max_id = type_info.max
        self.max_length = max_length

        self.memories = np.empty((0,),dtype=id_type)

    def add(self, input_memories:Union[int,list[int], np.ndarray,torch.Tensor], 
            is_sorted=True, is_duplicated=False) -> None:
        """
        Working Memoryに記憶を追加します。
        すでにWorking Memory がいっぱいのときはランダムに選ばれ、上書きされます。
        input_memoriesが昇順にsortいない場合はis_sorted=Falseにしてください。
        input_memoriesの中にに重複した記憶IDが存在する場合はis_duplicated=Trueにしてください

        Add input_memories to Working Memory.
        If the elements are added when the number of elements in W is maximum, 
        the elements in W are randomly chosen and overwritten.
        If input_memories is not sorted in ascending order, please set is_sorted=False
        If there are duplicate memory IDs in input_memories, please set is_duplicated=True
        """
        
        input_memories = self._type_check(input_memories)

        # fix duplication and sort.
        if is_duplicated:
            input_memories = np.unique(input_memories)
        elif not is_sorted:
            input_memories = np.sort(input_memories)

        # concatenation
        idxes = np.searchsorted(input_memories,self.memories)
        n_exist = np.logical_not(input_memories[idxes] == input_memories)
        mem = self.memories[n_exist]
        np.random.shuffle(mem)
        cut_len = self.max_length - len(input_memories)
        self.memories = np.concatenate([input_memories,mem[:cut_len]],axis=0)

    def _type_check(self,input_memories) -> np.ndarray:
        t = type(input_memories)
        if t is list:
            input_memories = np.array(input_memories,self.id_type)
        elif t is int:
            input_memories = np.array([t],self.id_type)
        elif t is torch.Tensor:
            input_memories = input_memories.detach().cpu().numpy()
        else:
            raise ValueError("Unknow id data type! {}".format(t))
        
        input_memories = input_memories.astype(self.id_type)
        return input_memories
        
    def create_pairs(self, duplicate=False) -> np.ndarray:
        """
        記憶辞書に登録するためのペアを作成します。
        duplicateがTrueの時は複数の記憶IDが一つの記憶に集まる場合があります。
            M_1 -> M_3
            M_2 -> M_3
        
        Create pairs to connect using Memory Dictionary.
        If duplicate is True, multiple memory IDs may be 
        connnected into a single memory.
            M_1 -> M_3
            M_2 -> M_3
        """

        srcs = self.memories.copy()
        if duplicate:
            tgt_idx = np.random.randint(0,len(self.memories),len(self.memories))
            tgts = self.memories[tgt_idx]
        else:
            tgts = self.memories.copy()
            np.random.shuffle(tgts)
        
        pairs = np.stack([srcs,tgts]).T
        return pairs

    def load_memories(self, memories:np.ndarray) -> None:
        """
        Working Memoryに記憶をロードします。
        loading memories to Working Memory.
        """
        assert len(memories)<= self.max_length
        dt = memories.dtype
        if self.id_type != dt:
            warning("loading memories dtype: {} is not self.id_type: {}! ".format(dt,self.id_type))
        self.memories = memories.astype(self.id_type)
        info("loaded memories")
