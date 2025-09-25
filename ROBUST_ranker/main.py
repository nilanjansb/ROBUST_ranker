import pandas as pd
import os
import argparse
import pyterrier as pt
from tqdm import tqdm
from ROBUST_ranker.modules.data import DataLoader
from ROBUST_ranker.modules.retrievers import Retrievers
from ROBUST_ranker import configuration
from pyterrier.measures import *
import pyterrier_alpha as pta

class RRanker():
    def __init__(self, **kwargs) -> None:

        # parse the command line 
        CLI=argparse.ArgumentParser()

        CLI.add_argument("--dataset", type=str)
        CLI.add_argument("--method", type=str, choices=['paraphrase', 'generic', 'specific'])
        CLI.add_argument("--variants", type=int)
        CLI.add_argument("--ranker", type=str, choices=['BM25','ColBERT','monoT5','SPLADE'])#,'SBERT','CE','SparseCE'
        CLI.add_argument("--depth", type=int, default=100)
        CLI.add_argument("--random_seed", type=int, default=42)
        CLI.add_argument("--verbose", type=bool, default=False)
        CLI.add_argument("--test", type=bool, default=False)

        args = CLI.parse_args()

        self.ds = args.dataset
        self.method = args.method
        self.variants = args.variants+1
        self.ranker = args.ranker
        self.depth = args.depth
        self.random_seed = args.random_seed
        self.verbose = args.verbose
        self.test = args.test
    
    def result_type_sync(self, res):
        try:
            res['qid'] = res['qid'].astype(str)
        except:
            pass
        try:
            res['docid'] = res['docid'].astype(int)
        except:
            res['docid'] = res['docno']
            res['docid'] = res['docid'].astype(int)
        try:
            res['docno'] = res['docno'].astype(str)
        except:
            res['docno'] = res['docid']
            res['docno'] = res['docno'].astype(str)
        try:
            res['rank'] = res['rank'].astype(int)
        except:
            pass
        try:
            res['score'] = res['score'].astype(float)
        except:
            pass
        try:
            res['query'] = res['query'].astype(object)
        except:
            pass

        try:
            res = res[['qid','docid','docno','rank','score','query','text']]
        except:
            res = res[['qid','docid','docno','rank','score','query']]
        return res

    def main(self):
        
        file_names, query_dfs = DataLoader.remove_symbols(self.ds,self.method,self.variants)
        if self.test:
            query_dfs = [query_dfs[0]] ## uncomment for testing
        # import the dataset
        dataset = pt.datasets.get_dataset(configuration.datasets[self.ds]['name'])
        qrels = dataset.get_qrels()

        model,index = Retrievers.load_model_index(self.ds,self.ranker)

        # run ranker on topics and the topic variants
        results = []
        print('Retrieval Started...')
        for topics in tqdm(query_dfs, total=len(query_dfs)):
            if self.ranker == 'BM25':
                res = Retrievers.retrieve_bm25( topics, index, self.depth)
            if self.ranker == 'ColBERT':
                res = Retrievers.retrieve_colbert(model, topics, index, self.depth)
            if self.ranker == 'monoT5':
                res = Retrievers.retrieve_monot5(model, topics, index, self.depth)
            if self.ranker == 'SPLADE':
                res = Retrievers.retrieve_splade(model, topics, index, self.depth)
            # if self.ranker == 'SBERT':
            #     res = Retrievers.retrieve_sbert(model, topics, index, self.depth)
            # if self.ranker == 'CE':
            #     res = Retrievers.retrieve_crossencoder(model, topics, index, self.depth)
            # if self.ranker == 'SparseCE':
            #     res = Retrievers.retrieve_crossencoder(model, topics, index, self.depth)
            
            # get text from lexical index if not present
            if 'text' not in res.columns:
                lmodel,lindex = Retrievers.load_model_index(self.ds,'BM25')
                tres = Retrievers.retrieve_doc_text(res, topics, lindex)
                tres = self.result_type_sync(tres)
                results.append(tres)
            else:
                results.append(res)
        print('======================================================================================')
        print(f'Retriever: {self.ranker}    Dataset:{self.ds}    Variant Type:{self.method}')
        print('======================================================================================')
        
        '''
        STORE RES FILES
        '''
        print('Storing RES files...')

        dirname = os.path.dirname(__file__)       
        flag_count = 0
        for res in results:
            file_name = f'{self.ranker}.{self.ds}.{self.method}.{flag_count}.tsv'
            file_loc = os.path.join(dirname, f'runs/text_res/{file_name}')
            res.to_csv(file_loc,index=False, sep = '\t')
            print('TEXT-RES File Saved: ', file_name)
            flag_count = flag_count+1

if __name__ == "__main__":
    RRanker().main()

'''
python -m ROBUST_ranker.main --dataset dl19 --method generic --variants 6 --ranker BM25 --depth 100
python -m ROBUST_ranker.main --dataset dl19 --method generic --variants 3 --ranker ColBERT --depth 100

python -m ROBUST_ranker.main --dataset dl19 --method generic --variants 1 --ranker BM25 --depth 100 --test True

HF_HUB_OFFLINE=1 \
python -m ROBUST_ranker.main --dataset dl19 --method paraphrase --variants 1 --ranker monoT5 --depth 100

'''