print('Loading Pyterrier')
import pyterrier as pt
# if not pt.started():
#     pt.init(version = 5.11, helper_version = "0.0.8")
print('Loading Pyterrier Complete')
import ast
from tqdm import tqdm
import pandas as pd
import pickle
import torch
from transformers import pipeline
from tqdm import tqdm
import argparse
from ROBUST_ranker import configuration
import os
import warnings
warnings.simplefilter(action='ignore', category=Warning)

class SimQ():
    def __init__(self,
                 **kwargs) -> None:
        
        # parse the command line 
        CLI=argparse.ArgumentParser()
        CLI.add_argument("--dataset", type=str)  
        CLI.add_argument("--loc", type=str, default='datasets/')  
        CLI.add_argument("--count", type=int, default=5) # maximum number of new queries to generate
        CLI.add_argument("--llm", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
        CLI.add_argument("--qtype", type=str, choices=['generic', 'specific', 'paraphrase', 'mixed'])
        CLI.add_argument("--token", type=str, default="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")
        CLI.add_argument("--test", type=bool, default=False)

        args = CLI.parse_args()

        self.loc = args.loc
        self.ds = args.dataset
        self.count = args.count
        self.llm = args.llm
        self.token = args.token
        self.qtype = args.qtype
        self.test = args.test

        self.pipe = pipeline(task="text-generation", model=self.llm, torch_dtype=torch.bfloat16, device_map='cuda:0',token=self.token)

    def remove_symbols(self, text):
        # initializing bad_chars_list
        bad_chars = [';',':','!',"*","/","?","'",'"',"-","_",".","%"]
        for i in bad_chars:
            text = text.replace(i, ' ')
        return str(text)

    def clean(self, queries):
        for i in tqdm(range(len(queries)), total=len(queries)):
            text = queries['query'][i]
            text = text.lower()
            text = self.remove_symbols(text)
            queries['query'][i] = text
        return queries
    
    def gen_sim_queries(self, query, count):
        if self.qtype=='paraphrase':
            system_prompt = f"Act as a linguistic expert specializing in semantic equivalence and stylistic variation. A query variant preserve the original meaning but differ in vocabulary, sentence structure, formality."
            user_prompt = f'Generate {count} distinct variant(s) of the question "{query}". Each variant should be in double quotes seperated using comma similar to a CSV format. Only output query variants and nothing else.'
        elif self.qtype=='mixed':
            system_prompt = f"You are a helpful and creative assistant. Generate query variants so as to make the information need expressed in the given query more generic or more specific. It is usually the case that adding terms makes a query more specific whereas removing a term makes it more general."
            user_prompt = f'Give me {count} {self.qtype} distinct variant(s) of the question "{query}". Each variant should be in double quotes seperated using comma similar to a CSV format. Only output query variants and nothing else.'
        else:
            system_prompt = f"You are a helpful and creative assistant. Generate query variants so as to make the information need expressed in the given query more generic or more specific. It is usually the case that adding terms makes a query more specific whereas removing a term makes it more general."
            user_prompt = f'Give me {count} {self.qtype} distinct variant(s) of the question "{query}". Each variant should be in double quotes seperated using comma similar to a CSV format. Only output query variants and nothing else.'

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.pipe(messages, pad_token_id=self.pipe.tokenizer.eos_token_id)
        s = response[0]['generated_text'][2]['content']
        parsed = s.split('", "')
        return parsed

    def get_data(self, ds):
        dataset = pt.datasets.get_dataset(configuration.datasets[ds]['name'])
        if ds=='fair22.eval':
            cwd = os.getcwd()
            topics = pt.io.read_topics(cwd+'/LLM_ranker/datasets/topics_qrels/fair22.eval.topics.txt', format='singleline')
            qrels = pt.io.read_qrels(cwd+'/LLM_ranker/datasets/topics_qrels/fair22.eval.qrels.txt')
        else:
            topics = dataset.get_topics(configuration.datasets[ds]['topics'])
            qrels = dataset.get_qrels()
        
        topics = self.clean(topics)

        return dataset,topics,qrels
    
    def merge_in_single_tsv(self, ds, method, variants):
        file_names = []

        for v_no in range(0,variants+1):
            name = f'ROBUST_ranker/datasets/sim_queries_Meta-Llama-3-8B-Instruct/{ds}_{method}_{v_no}.tsv'
            file_names.append(name)

        query_dfs = []

        for file_name in file_names:
            num = file_name.split('_')[-1].split('.')[0]
            #print(num)
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

    def main(self):

        dirname = os.path.dirname(__file__)

        dataset,topics,qrels = self.get_data(self.ds)

        if self.test: # check two queries during testing
            print('WARNING: Testing mode is set to use 2 queries.')
            topics = topics.head(2)

        final_sim_topics = pd.DataFrame(columns=['qid','query','sim_query'])
        print('Generating Query Variants...')
        for index,row in tqdm(topics.iterrows(),total=len(topics)):
            qid = row['qid']
            query_text = topics['query'].loc[topics['qid']==str(qid)].values[0]
            gens = self.gen_sim_queries(query_text,self.count)
            ## getting all the variants
            len_gens = len(gens)
            qids = [qid]*len_gens
            query_texts = [query_text]*len_gens

            final_gens = []
            for gen in gens:
                gen_text = self.remove_symbols(gen)
                final_gens.append(gen_text.lower())
            
            output_rows = {'qid':qids, 'query':query_texts, 'sim_query':final_gens}

            output_df = pd.DataFrame(output_rows)

            final_sim_topics = pd.concat([final_sim_topics, output_df], axis=0)

        sim_topics = final_sim_topics.copy()

        print('Writing Query Variants...')

        # store the file
        dirname = os.path.dirname(__file__)
        llm_name = self.llm.split('/')[-1]
        dir_name = f'sim_queries_{llm_name}/'

        # now create seperate query files with each variant
        for ver_num in range(0,self.count+1):
            # create the current query file
            topic_varint = topics.copy()
            if ver_num!=0: # the 0-th variant is the original query
                for index,row in tqdm(topic_varint.iterrows(),total=len(topic_varint)):
                    qid = row['qid']
                    try:
                        if len(sim_topics['sim_query'].loc[sim_topics['qid']==str(qid)].values[0].split(','))==self.count:
                            query_variant_text = sim_topics['sim_query'].loc[sim_topics['qid']==str(qid)].values[0].split(',')[ver_num-1]
                        else:
                            query_variant_text = sim_topics['sim_query'].loc[sim_topics['qid']==str(qid)].values[ver_num-1]
                        topic_varint.loc[index, 'query'] = query_variant_text
                    except:
                        print('.')
            
            file_name = f'{self.ds}_{self.qtype}_{(ver_num)}.tsv'

            try:
                dir_loc = store_loc = os.path.join(dirname, self.loc, dir_name)
                os.mkdir(dir_loc)
                print(f'Directory created: {dir_loc}')
            except:
                pass

            store_loc = os.path.join(dirname, self.loc, dir_name, file_name)
            

            topic_varint.to_csv(store_loc, sep='\t', index=False, header=True)

        self.merge_in_single_tsv(self.ds, self.qtype, self.count)

if __name__ == "__main__": 
    SimQ().main()


'''
 python -m ROBUST_ranker.simq --dataset dl19 --qtype generic --count 5 --test False
'''
