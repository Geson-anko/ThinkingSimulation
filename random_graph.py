"""\
このファイルはランダムグラフを生成し、Adjacency matrixと
描画したグラフを保存するスクリプトを提供します。

Usage:
    settingファイルを作成し、ターミナルから次のコマンドを実行してください。
    $ python random_graph.py path/to/settings.json

settingの形式:
    JSON形式で記述してください。記述例は次のファイルを参照してください。
    settings/random_graph_example.json

    設定可能項目:
        num_nodes: int  ノード数です。
        connection_probs: List[int] ノード接続確率です。このリストの要素数の数だけグラフを生成します。
        directed: bool  グラフのエッジに方向を持たせるか否です。
        self_connection:str 自己結合の設定です。"all", "no", "allow" のいずれかを設定できます。
            "all":  すべてのノードは自己結合をします。Adjacency matrixの対角はすべて1です。
            "no":   すべてのノードは自己結合をしません。Adjacency matrixの対角はすべて0です。
            "allow":connection_probsの確率にしたがって自己結合します。
        outdir: str グラフデータを保存する親ディレクトリです。
        suffix: str グラフを保存するディレクトリです。
        
        グラフは次のような形でフォルダに保存されます。
        outdir/n{num_node}_p{prob}_di{directed}_sc{self_connection}{suffix}/plot.png, ...
"""
import numpy as np
from graph_tools.constants import SC

def random_generate(num_nodes:int, prob:float, directed:bool, sc:str) -> np.ndarray:
    """
    結合確率probのランダムグラフのadjacency matrixを生成します。
    Args:
        num_nodes: 最大ノード数です。
        prob: 結合確率です。
        directed: エッジの方向の有無です。
        sc: self_connection の略で、自己結合の有無を決定します。
            "all", "no", "allow" のいずれかです。
    """
    assert 0.0 < prob < 1.0
    if directed:
        adj_mat = generate_directed(num_nodes,prob)
    else:
        adj_mat = generate_undirected(num_nodes, prob)
    
    # self connection process
    indices = np.arange(num_nodes)
    if sc == SC.ALL:
        adj_mat[indices,indices] = True
    elif sc == SC.NO:
        adj_mat[indices,indices] = False
    elif sc == SC.ALLOW:
        pass
    else:
        raise ValueError(f"{sc} is invalid self-connection option! please use all, no, or allow.")

    return adj_mat

def generate_directed(num_nodes:int, prob:float) -> np.ndarray:
    """向きのある接続確率probのランダムグラフのadajcency matricを返します。"""
    p,q = prob, 1 - prob
    adj_mat = np.random.choice([True,False],num_nodes**2, replace=True, p=[p,q])
    return adj_mat

def generate_undirected(num_nodes:int, prob:float) -> np.ndarray:
    """向きのない接続確率probのランダムグラフのadajcency matricを返します。"""
    p,q = prob, 1 - prob
    adj_mat = np.zeros((num_nodes,num_nodes),bool)
    indices = np.arange(num_nodes)
    for i in range(num_nodes):
        TF = np.random.choice([True,False], num_nodes - i, True, p=[p,q])
        adj_mat[i,indices[i:]] = TF
        adj_mat[indices[i:],i] = TF

    return adj_mat


if __name__ == "__main__":
    from argparse import ArgumentParser
    from attrdict import AttrDict
    import json
    import os
    from utils import load_setting
    # arguments
    parser = ArgumentParser()
    parser.add_argument("setting_file",type=str)
    args = parser.parse_args()
    
    # load settings
    setting = load_setting(args.setting_file)

    num_graphs = len(setting.connection_probs)
    num_nodes = setting.num_nodes
    connection_probs = setting.connection_probs
    directed = setting.directed
    self_connection = setting.self_connection
    out_par_dir = setting.outdir
    suffix = setting.suffix

    print("Generating random graphs...")
    for i in range(num_graphs):
        p = connection_probs[i]

        # output paths
        out_dir = f"{out_par_dir}/n{num_nodes}_p{p}_di{directed}_sc{self_connection}{suffix}"
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        plots2d_path = f"{out_dir}/2dplots.png"
        ajace_mat_path = f"{out_dir}/ajacency_mat.npz"

        adjacency_mat = random_generate(num_nodes, p, directed, self_connection)
        pairs = adjac2pairs(adjacency_mat)
        G = pairs2graph(pairs, directed, num_nodes=num_nodes)
        save_graph(G,plots2d_path)
        np.save(adjacency_mat, ajace_mat_path)
    print("Finished.")

