__doc__ = """\
This class is a abstract class of Memory Dictionary.
記憶辞書を実装するためのいくつかの重要なメソッドが定義されています。
記憶辞書を実装するときは、次のメソッドを必ず実装してください。

- __init__(self, num_memory:int, num_dims:int,*, device="cpu", dtype=torch.float) -> None:

- connect(self, src_ids:torch.Tensor, tgt_ids:List[torch.Tensor]) -> None:

- trace(self, src_ids:torch.Tensor) -> torch.Tensor:

- add_memories(self, num:int) -> None:

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
                [True,True,True,True,True]
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
