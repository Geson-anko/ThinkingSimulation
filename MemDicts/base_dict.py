__doc__ = """\
This class is a abstract class of Memory Dictionary.
記憶辞書を実装するためのいくつかの重要なメソッドが定義されています。
記憶辞書を実装するときは、次のメソッドを必ず実装してください。

- __init__(self, num_memory:int, num_dims:int,*, device="cpu", dtype=torch.float) -> None:

- connect(self, src_ids:torch.Tensor, tgt_ids:List[torch.Tensor]) -> None:

- trace(self, src_ids:torch.Tensor) -> torch.Tensor:

- add_memories(self, num:int) -> None:

- get_memory_vector(self, src_ids:torch.Tensor) -> None:

"""
USAGE = """\
記憶辞書の使い方
0. 形式
記憶辞書は各記憶に対して 0 から順番にIDを与えて管理します。
将来的に使われる記憶ベクトルのバッファを確保し、計算対象から除外する機能はありません。

1. 記憶の個数と、記憶ベクトルの次元を指定して、コンストラクトしてください。
    >>> d = MemDict(num_memory, num_dims)

2. 辞書に記憶(src_ids) とその検索結果 (tgt_ids) を登録します。
    >>> d.connect(src_ids, tgt_ids) # or d[src_ids] = tgt_ids

    src_idsとtgt_idsは形式が異なるためご注意ください。

    - src_idsのフォーマット
        1d 重複なしの 整数型 配列です。各要素は 0 以上 num_memory 未満です。
            Ex. 
            >>> src_ids = [9,3,4,2,1] # num_memory = 10

    - tgt_ids のフォーマット
        src_idsの各要素につながった記憶IDの配列を持った、リストです。
        要素数はsrc_idsと等しいです。
            Ex. 
            >>> src_ids = [1,2]
            >>> tgt_ids = [
                [1,0,3,4,2],
                [3,2,0]
            ]
        内部で繋がっているindexをTrueに、そうでない場合はFalseにした2D bool Tensorに
        変換されるため、直接 2d bool Tensorを入力しても良いです。
        (内部で tgt_ids = d.format_tgt_ids(tgt_ids) を実行しています。)
        その場合テンソルの形状は次の形でなければなりません。
            Ex.
            >>> tgt_ids.shape
            torch.Size([len(src_ids), num_memory])
            >>> src_ids = [1,2] # num_memory=5
            >>> tgt_ids = [ # 上記と同じ
                [True,True,True,True,True],
                [True,False,True,True,False]
            ]

3. 辞書をたどり、記憶を取り出します。
    >>> d.trace(src_ids) # or d[src_ids]
    connected_ids


    trace メソッドの出力は重複のない 1d 整数型 配列です。   
    よって、src_idsの中のidそれぞれに対してつながった記憶を取り出す場合は,
    次のメソッドを仕様してください。
    >>> d.trace_each(src_ids) 
    List[torch.Tensor]


4. 記憶を追加します。
    >>> d.add_memories(num)

    記憶辞書の扱うIDの数を増やします。
    内部で次の処理が行われています。
    >>> num_memory += num

5. 記憶ベクトルを取得します。
    >>> d.get_memory_vector(src_ids)
    torch.Tensor

    src_idsの中のidそれぞれに対して記憶ベクトルを取得します。
    return valueは2DのTensorです。
"""
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import *

class _memory_dictionary(nn.Module, metaclass=ABCMeta):
    __doc__ = """\
        このクラスは記憶辞書の抽象クラスです。
        {}
        """.format(USAGE)

    @abstractmethod
    def __init__(
        self, num_memory:int, num_dims:int, *,
        device:torch.device=None, dtype:torch.float=None
        ) -> None:
        super().__init__()
        self.num_memory = num_memory
        self.num_dims = num_dims

    @abstractmethod
    def connect(self, src_ids:torch.Tensor, tgt_ids:Union[List[torch.Tensor], torch.Tensor]) -> None:
        assert len(src_ids) == len(tgt_ids)
        return self.format_tgt_ids(tgt_ids)
        
    @abstractmethod
    def trace(self, src_ids:torch.Tensor) -> torch.Tensor:
        pass

    def trace_each(self, src_ids:torch.Tensor) -> torch.Tensor:
        result = [self.trace(i.view(1)) for i in src_ids]
        return result
    
    @abstractmethod
    def add_memories(self,num:int ) -> None:
        self.num_memory += num
        
    @abstractmethod
    def get_memory_vector(self, src_ids:torch.Tensor) -> torch.Tensor:
        pass

    def format_tgt_ids(self, tgt_ids:Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        t = type(tgt_ids)
        if t is list:
            l = len(tgt_ids)
            result = torch.zeros(l, self.num_memory, dtype=torch.bool)
            for i,ids in enumerate(tgt_ids):
                result[i][ids] = True

        elif t is torch.Tensor:
            dt = tgt_ids.dtype
            if not dt is torch.bool:
                raise ValueError(f"Please bool tensor. Input tensor dtype is {dt}")
            assert tgt_ids.size(1) == self.num_memory
            
            result = tgt_ids
        else:
            raise ValueError(f"Please list of ids or torch.Tensor, Unknow input type: {t}.")

        
        return result




