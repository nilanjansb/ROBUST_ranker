import pandas as pd
import os
import argparse
from tqdm import tqdm
from ROBUST_ranker import configuration
from umbrela.gpt_judge import GPTJudge
from dotenv import load_dotenv
import json
import glob

class GenRelLabels():
    def __init__(self, **kwargs) -> None:

        # parse the command line 
        CLI=argparse.ArgumentParser()

        CLI.add_argument("--dataset", type=str)
        CLI.add_argument("--method", type=str, choices=['paraphrase', 'generic', 'specific'])
        CLI.add_argument("--variants", type=int)
        CLI.add_argument("--ranker", type=str, choices=['BM25','ColBERT','monoT5','SPLADE','SBERT','CE','SparseCE'])
        CLI.add_argument("--depth", type=int, default=10)
        CLI.add_argument("--random_seed", type=int, default=42)
        CLI.add_argument("--verbose", type=bool, default=False)
        CLI.add_argument("--test", type=bool, default=False)
        CLI.add_argument("--model_name", type=str, default='gpt-4o-mini')
        CLI.add_argument("--few_shot_count", type=int, default=0)
        args = CLI.parse_args()

        self.ds = args.dataset
        self.method = args.method
        self.variants = args.variants
        self.ranker = args.ranker
        self.depth = args.depth
        self.random_seed = args.random_seed
        self.verbose = args.verbose
        self.test = args.test
        self.model_name = args.model_name
        self.few_shot_count = args.few_shot_count

        load_dotenv()

        # our method
        #self.judge_gpt = GPTJudge(qrel=f"{self.ds}-passage", prompt_type="variant", model_name=self.model_name, few_shot_count=self.few_shot_count)
        
        # umbrela method
        self.judge_gpt = GPTJudge(qrel=f"{self.ds}-passage", prompt_type="bing", model_name=self.model_name, few_shot_count=self.few_shot_count)
    
    def create_input_dict(self, res, topics):
        qids = res['qid'].unique()

        input_dicts = []

        for qid in tqdm(qids, total=len(qids)):
            sub_res = res.loc[res['qid']==qid]
            sub_res = sub_res.head(self.depth) # depth cutoff
            query_text = sub_res['query'].values[0]
            original_query_text = topics['query'].loc[topics['qid']==qid].values[0]

            candidates = []

            for index,row in sub_res.iterrows():
                docid = row['docid']
                segment = row['text']
                candidate = {
                    "doc":{
                        "segment": segment
                    },
                    "docid":str(docid)
                }
                candidates.append(candidate)

            input_dict = {
                "query": {
                    "original": original_query_text,
                    "text":query_text,
                    "qid": str(qid),
                    "variant": self.method
                },
                "candidates": candidates
            }

            input_dicts.append(input_dict)
            
        return input_dicts
    
    def generate_labels(self, input_dicts):
        gen_qrels = []
        for input_dict in tqdm(input_dicts, total=len(input_dicts)):
            qid = input_dict['query']['qid']
            doc_info = input_dict['candidates']
            
            judgments = self.judge_gpt.judge(request_dict=input_dict)
            
            for a,b in zip(doc_info,judgments):
                cur_qrel = {'qid':qid,'docno':a['docid'], 'label':b['judgment'], 'iteration': 'Q0','prediction':b['prediction'], 'result_status':b['result_status']}
                gen_qrels.append(cur_qrel)
        
        final_qrels = pd.DataFrame(gen_qrels)
        return final_qrels
    
    def main(self):
        print('======================================================================================')
        print(f'Retriever: {self.ranker}    Dataset:{self.ds}    Variant Type:{self.method}')
        print('======================================================================================')

        '''
            LOAD RES FILES
        '''
        print('Loading RES files...')

        dirname = os.path.dirname(__file__)
        text_res_dir = os.path.join(dirname, f'runs/text_res/')

        

        final_files = glob.glob(f'{text_res_dir}{self.ranker}.{self.ds}.{self.method}.*')
        final_files.sort()

        print(final_files)
        avail_variants = []
        text_ress = []
        
        ### Add function to skip generating when QREL label for query and doc is already available. *** Then start generating qrels

        for final_file in final_files:
            var = final_file.split('/')[-1].split('.')[-2]
            avail_variants.append(var)

            if var=='0':
                text_res = pd.read_csv(final_file,sep='\t')
                topics = text_res[['qid','query']]

            if var!='0': ## Skipping the original query QRELS generation
            
                text_res_labeled_dir = os.path.join(dirname, f'runs/gen_qrels/')
                qrels_file_path = f'{text_res_labeled_dir}{self.ranker}.{self.ds}.{self.method}.{self.model_name}.{self.few_shot_count}S.{var}.qrels'

                '''
                    CHECK IF GENERATED QRELS ARE AVAILABLE
                '''
                if os.path.exists(qrels_file_path):
                    print('QRELS for the variant exist: ', qrels_file_path.split('/')[-1])
                else:
                    print('File Does Not Exist. Generating QRELS for the variant: ', qrels_file_path.split('/')[-1])
                    text_res = pd.read_csv(final_file,sep='\t')
                    print(var)
                    
                    input_dicts = self.create_input_dict(text_res, topics)
                    gen_qrels = self.generate_labels(input_dicts)

                    gen_qrels.to_csv(qrels_file_path,index=False, sep = '\t')
                    print('File Saved: ', qrels_file_path.split('/')[-1])

if __name__ == "__main__": 
    GenRelLabels().main()

'''
python -m ROBUST_ranker.generate_relevance_labels --dataset dl19 --method paraphrase --ranker BM25 

python -m ROBUST_ranker.generate_relevance_labels --dataset dl19 --method specific --ranker BM25 

python -m ROBUST_ranker.generate_relevance_labels --dataset dl19 --method generic --ranker BM25


'''