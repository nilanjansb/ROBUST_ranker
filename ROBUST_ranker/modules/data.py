import pandas as pd
import os
from ROBUST_ranker.modules.retrievers import Retrievers
class DataLoader():
    def remove_symbols(ds, method, v_nos):
        # initializing file names
        file_names = []
        cwd = os.getcwd()
        for v_no in range(0,v_nos):
            name = f'{cwd}/ROBUST_ranker/datasets/sim_queries_Meta-Llama-3-8B-Instruct/{ds}_{method}_{v_no}.tsv'
            file_names.append(name)

        query_dfs = []

        # load the topics
        for file_name in file_names:
            num = file_name.split('_')[-1][0]
            cur_df = pd.read_csv(file_name, sep='\t')
            cur_df = Retrievers.clean(cur_df)
            query_dfs.append(cur_df)

        return file_names, query_dfs