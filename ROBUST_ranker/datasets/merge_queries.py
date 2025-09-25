import pandas as pd

ds = 'dl19'
method = 'generic'
variants = 20

file_names = []

for v_no in range(0,variants+1):
    name = f'ROBUST_ranker/datasets/sim_queries_from_llm/{ds}_{method}_{v_no}.tsv'
    file_names.append(name)

query_dfs = []

for file_name in file_names:
    num = file_name.split('_')[-1].split('.')[0]
    print(num)
    if num=='0':
        cur_df = pd.read_csv(file_name, sep='\t')
        cur_df = cur_df.rename(columns={'query': 'original_query'})
    else:  
        cur_df = pd.read_csv(file_name, sep='\t', usecols=['query'])
        cur_df = cur_df.rename(columns={'query': f'query_variant_{num}'})
    query_dfs.append(cur_df)

merged_df = pd.concat(query_dfs,axis=1)

f_name = f'ROBUST_ranker/datasets/{ds}_{method}.tsv'
merged_df.to_csv(f_name, index=False, sep='\t')