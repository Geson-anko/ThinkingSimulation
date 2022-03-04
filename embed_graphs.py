"""\
このスクリプトでは、グラフのadajcency matrixを記憶辞書に埋め込み、
以下の結果を保存します。
    - 記憶辞書のパラメータ 
    - 再構成されたadajacency matrix
    - 再構成されたadjacency matrix を2d graphにプロットした画像
    - 各記憶ベクトルのmatrix
    - tensorboardに各記憶ベクトルを3D投影した結果
    - その他メトリクス

Usage:
    settingファイルを作成し、ターミナルから次のコマンドを実行してください。
    $ python embed_graphs.py --setting_file=path/to/settings.json
    gpuで実行する場合は、 --num_gpus=1のオプションをつけてください。

settingの形式:
    JSON形式で記述してください。記述例は次のファイルを参照してください。
    settings/cossim_example.json

    設定項目:
        num_dims: int   埋め込み次元数です。
        graphs_dir: str 使用するgraphのディレクトリです。1つのグラフは以下のように配置していてください。
            graphs_dir/**/adjacency_mat.npz 
        lrs: list[float] 学習率の配列です。
            adjacency_matrixの数 x len(lrs)の数の結果が出力されます。
        out_dir: str 結果を保存するディレクトリです。
        dict_type: str 使用する辞書の型です。

        辞書のインスタンス時に必要なオプションの設定も書いてください。
        Ex.
            stochastic: bool CosSimMemDictを使用する際のオプションです。

        結果は次のような形式でフォルダに保存されます。
            {our_dir}/dt{dict_type}_lr{lr} + 辞書の形式ごとのオプション/
                adjacency_matの親ディレクトリ名/params.pth ...
"""
import numpy as np
import torch
from utils import load_setting
from graph_tools.io import get_adjacency_mat_paths, get_graph_names,save_graph
from graph_tools.convertion import tracedall2adjac, adjac2pairs, pairs2graph
from graph_tools.metrics import accuracy, extra, shortage
from torch.utils.tensorboard import SummaryWriter

class DictType:
    COSSIM = "cossim"

def main(args):
    # load_settings
    setting = load_setting(args.setting_file)
    num_dims = setting.num_dims
    graph_dir = setting.graph_dir
    lrs = setting.lrs
    out_dir = setting.out_dir
    dict_type = setting.dict_type

    graph_paths = get_adjacency_mat_paths(graph_dir)
    graph_names = get_graph_names(graph_paths)
                
    # checking dict_type
    if dict_type == DictType.COSSIM:
        from MemDicts.cossim import CosSimMemDict as MemDict
    else:
        raise ValueError("Use cossim for dict_type. unknow dict_type: {}".format(dict_type))



    # set device
    ng = args.num_gpus
    if ng == 0:
        device = "cpu"
    else:
        device = "cuda"
    print("Using device:",device)

    num_roops = len(lrs)
    num_graphs = len(graph_paths)
    print(f"num_roop: {num_roops}, num_graphs: {num_graphs}, all: {num_graphs*num_roops}")
    print("Embedding...")
    
    for i in range(num_roops):
        for j in range(num_graphs):
            lr = lr[i]
            gn = graph_names[j]

            # load adjacency matrix
            adj_mat = np.load(graph_paths[j])
            adj_mat = torch.from_numpy(adj_mat).to(device)
            num_nodes = len(adj_mat)

            # instance and option str
            if dict_type == DictType.COSSIM:
                # opt str
                stc = args.stochastic
                if stc:
                    opt_str = f"stc{stc}"
                    thres = 0.0
                else:
                    thres = args.threshold
                    opt_str = f"thres{thres}"

                # instance
                mem_dict = MemDict(num_nodes,num_dims, device=device, lr=lr, stochastic=stc, threshold=thres)
            ##########################

            # connect
            src_ids = torch.arange(num_nodes)
            mem_dict.connect(src_ids,adj_mat)

            # trace each
            output = mem_dict.trace_each(src_ids)

            # to adjacency matrix
            rec_adj_mat = tracedall2adjac(output)

            # save dir
            save_dir = f"{out_dir}/dt{dict_type}_lr{lr}_{opt_str}/{gn}"
            print("save to",save_dir)

            # calc metrics 
            acc = accuracy(adj_mat,rec_adj_mat)
            ext = extra(adj_mat,rec_adj_mat)
            sht = shortage(adj_mat,rec_adj_mat)
            s = f"accuracy: {acc:3.2f}%, extra: {ext:3.2f}%, shortage: {sht:3.2f}%"
            with open(f"{save_dir}/metrics.txt","w",encoding="utf-8") as f:
                f.write(s)

            # save memory vectors
            mp = f"{save_dir}/memory_vectors.pth"
            memory_vectors = mem_dict.get_memory_vector(src_ids)
            torch.save(memory_vectors,mp.cpu())

            # 3D embedding
            writer = SummaryWriter(save_dir)
            writer.add_embedding(memory_vectors,src_ids.numpy())
            writer.close()

            # 2d plotting
            gp = f"{save_dir}/rec_2dplots.png"
            pairs = adjac2pairs(rec_adj_mat)
            G = pairs2graph(pairs,True,num_nodes)
            save_graph(G,gp)

            # save weight
            pp = f"{save_dir}/params.pth"
            torch.save(mem_dict.state_dict(),pp)
            
if __name__ == "__main__":
    from argparse import ArgumentParser 
    
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--setting_file",type=str,default="settings/cossim_example.json")
    parser.add_argument("--num_gpus",type=int,default=0)
    args = parser.parse_args()
    main(args)
        



