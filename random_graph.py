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
class SC:
    NO = "no"
    ALL = "all"
    ALLOW = "allow"