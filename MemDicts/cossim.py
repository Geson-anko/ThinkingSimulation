from .base_dict import _memory_dictionary, USAGE
import torch
from typing import *

class CosSimMemDict(_memory_dictionary):
    __doc__ = """\
        コサイン類似度を用いた記憶辞書です。

        {}
        """.format(USAGE)
    def __init__(
        self, num_memory: int, num_dims: int, *, 
        device: torch.device = None, dtype: torch.float = None,
        lr:float = 1.0, stochastic:bool = True, threshold:float = 0.8
    ) -> None:
        super().__init__(num_memory, num_dims)
        """
        Key words:
            lr:     学習率です。大きいほど辞書更新時にベクトルが大きく更新されます。
            stochastic: 確率的に記憶を取り出すか、否かです。
            threshold:  stochastic がFalseのときに使われます。記憶の接続の有無を決める閾値です。

        各記憶に対応したベクトルを用意します。
        weightの形状は num_memory x num_dimsです。
        初期化時は 平均0, 分散1 の標準分布に従い、各記憶ベクトルは
        traceのときのコサイン類似度の計算を軽量にするため、ノルムで割られます。
        """
        
        self.lr = lr
        self.stocastic = stochastic
        self.threshold = threshold
        self.memory_indices = torch.arange(self.num_memory, dtype=torch.long)
        mv = torch.randn((num_memory, num_dims),device=device, dtype=dtype)
        mv = mv / torch.linalg.norm(mv,dim=1,keepdim=True)
        self.weight = torch.nn.Parameter(mv, False)
        

    def connect(self, src_ids: torch.Tensor, tgt_ids: Union[List[torch.Tensor], torch.Tensor]) -> None:
        cons =  super().connect(src_ids, tgt_ids)
        """
        誤差逆伝搬や微分を行わずに辞書を更新します。
        コサイン類似度を最大化するのではなく、内積を最大化する問題として勾配行列を求めます。
        そして更新後にその更新対象のベクトルをノルムで割ります。
        変化するのは、src_idsのベクトルです。
        connect処理は並列に行われます。
        """
        src_vecs = self.weight[src_ids]
        sign = (-1) ** cons.type_as(self.weight)
        grad_vecs = torch.matmul(sign, self.weight) / self.num_memory
        updated = src_vecs - self.lr * grad_vecs
        updated = updated / torch.linalg.norm(updated,dim=1, keepdim=True)
        self.weight.data[src_ids] = updated

    def trace(self, src_ids: torch.Tensor) -> torch.Tensor:
        """
        src_idsに繋がった記憶を取り出します。
        各記憶ベクトルと内積を計算します。記憶ベクトルは初期化時や更新時に
        ノルムを1にしているため、コサイン類似度と一致します。
        stocasticがTrueの場合は、内積を0 ~ 1 の範囲にclampし、
        大きいほど取り出されやすくなります。
        そうでない場合は内積がthresholdよりも大きい場合に取り出されます。
        """
        cons = torch.zeros(self.num_memory, device=bool)
        for i, sid in enumerate(src_ids):
            out = torch.matmul(self.weight, self.weight[sid])
            if self.stocastic:
                out_ids = self.get_with_stocastic(out)
            else:
                out_ids = self.get_with_threshold(out)

            cons[out_ids] = True
        memory = self.memory_indices[cons]
        return memory

    def get_with_stocastic(self, out:torch.Tensor) -> torch.Tensor:
        """
        outを0 ~ 1 の範囲にclampし、コイン投げをしてシュル直するIDを決めます。
        """
        true_prob = torch.clamp(out, 0,1.0)
        false_prob = 1 - true_prob
        prob = torch.stack([false_prob,true_prob]).T
        TF = torch.multinomial(prob,1).view(-1).bool()
        return self.memory_indices[TF]

    def get_with_threshold(self, out:torch.Tensor) -> torch.Tensor:
        """
        outの要素の値がthresholdよりも大きい場合は、それに対応する記憶のidxを返します。
        """
        TF = out > self.threshold
        return self.memory_indices[TF]


    def add_memories(self, num: int) -> None:
        """
        numだけ記憶ベクトルを増やします。
        """
        new = torch.randn(num, self.num_dims).type_as(self.weight)
        new = new / torch.linalg.norm(new, dim=1,keepdim=True)
        new_ids = torch.arange(self.num_memory, num + self.num_memory,dtype=torch.long)
        self.memory_indices = torch.cat([self.memory_indices, new_ids])
        self.weight.data = torch.cat([self.weight.data, new],0)

        super().add_memories(num)

    
    def get_memory_vector(self, src_ids: torch.Tensor) -> torch.Tensor:
        """
        src_idsの要素それぞれに対応した記憶ベクトルを返します。
        """
        return self.weight[src_ids]

        


    

