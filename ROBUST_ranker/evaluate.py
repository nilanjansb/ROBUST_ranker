import pandas as pd
import os
import argparse
from tqdm import tqdm
from ROBUST_ranker import configuration
import pyterrier as pt
from pyterrier.measures import *
import pyterrier_alpha as pta
import glob
import statistics
from ROBUST_ranker.main import RRanker
from evaluate import load

class Evaluation():
    def __init__(self, **kwargs) -> None:

        # parse the command line 
        CLI=argparse.ArgumentParser()

        CLI.add_argument("--dataset", type=str)
        CLI.add_argument("--method", type=str, choices=['paraphrase', 'generic', 'specific'])
        CLI.add_argument("--variants", type=int)
        CLI.add_argument("--ranker", type=str, choices=['BM25','ColBERT','monoT5','SPLADE','SBERT','CE','SparseCE'])
        CLI.add_argument("--depth", type=int, default=100)
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
        self.bertscore = load("bertscore")
        self.few_shot_count = args.few_shot_count

    def list_avg(self, lst):
        try:
            lst = lst[1:]
            avg = sum(lst)/len(lst)
        except:
            avg=-1
        return avg

    def evaluate_res_qrels(self, results, qrels_list, version_name, reference_queries):

        query_type = []
        ap = []
        nd10 = []
        rbo = []
        bscores = []
        rel_measures = [AP(rel=2), NDCG(cutoff=10), pta.RBO(results[0],p=0.9)]
        
        for res,qrels,version in zip(results,qrels_list,version_name):
            
            # print(res.dtypes)
            eval_score = pt.Evaluate(res,qrels,metrics=rel_measures)
            query_type.append(f'Variant {version}')
            ap.append(round(eval_score['AP(rel=2)'],4))
            nd10.append(round(eval_score['nDCG@10'],4))
            rbo.append(round(eval_score['RBO(p=0.9)'],4))
            if len(reference_queries)>0:
                generated_queries = list(res['query'].unique())
                bscore = self.bertscore.compute(predictions=generated_queries, references=reference_queries, lang="en")
                bscores.append(round(statistics.mean(bscore['f1']),4))

        if len(reference_queries)>0:
            res_df = pd.DataFrame({'Query':query_type, 'AP(rel=2)': ap, 'nDCG@10': nd10, 'RBO(p=0.9)':rbo,'BERT_Score':bscores})
        else:
            res_df = pd.DataFrame({'Query':query_type, 'AP(rel=2)': ap, 'nDCG@10': nd10, 'RBO(p=0.9)':rbo})
        return res_df
        
    def main(self):

        # import the dataset
        dataset = pt.datasets.get_dataset(configuration.datasets[self.ds]['name'])
        qrels = dataset.get_qrels()

        print('Loading RES files that have QRELS...')

        dirname = os.path.dirname(__file__)
        text_res_dir = os.path.join(dirname, f'runs/text_res/')

        final_files_path = glob.glob(f'{text_res_dir}{self.ranker}.{self.ds}.{self.method}.*')
        avail_variants = []
        final_files_path.sort()
        qrels_list_original = []
        qrels_list_generated = []
        version_name = []
        results = []

        for final_file_path in final_files_path:
            var = final_file_path.split('/')[-1].split('.')[-2]
            text_res_labeled_dir = os.path.join(dirname, f'runs/gen_qrels/')
            qrels_file_path = f'{text_res_labeled_dir}{self.ranker}.{self.ds}.{self.method}.{self.model_name}.{self.few_shot_count}S.{var}.qrels'

            if var=='0':
                qrels_list_generated.append(qrels)
                text_res = pd.read_csv(final_file_path,sep='\t')
                text_res = RRanker.result_type_sync(self, text_res)
                results.append(text_res)
                qrels_list_original.append(qrels)
                version_name.append(var)
                reference_queries = list(text_res['query'].unique())

                
            '''
                CHECK IF GENERATED QRELS ARE AVAILABLE
            '''
            if os.path.exists(qrels_file_path):
                print('QRELS for the variant exist: ', qrels_file_path.split('/')[-1])

                # load results
                text_res = pd.read_csv(final_file_path,sep='\t')
                results.append(text_res)
                
                # load original qrels
                #print(qrels)
                qrels_list_original.append(qrels)
                gen_qrels = pd.read_csv(qrels_file_path,sep='\t', usecols = ['qid','docno','label'])
                
                # format qrels
                if 'iteration' not in gen_qrels.columns:
                    gen_qrels['iteration'] = ['Q0']*len(gen_qrels)
                gen_qrels['qid'] = gen_qrels['qid'].astype(str)
                gen_qrels['qid'] = gen_qrels['qid'].astype(object)
                gen_qrels['docno'] = gen_qrels['docno'].astype(str)
                gen_qrels['docno'] = gen_qrels['docno'].astype(object)

                qrels_list_generated.append(gen_qrels)
                version_name.append(var)
            else:
                if var!='0':
                    print('File Does Not Exist. Generate QRELS and retry: ', qrels_file_path.split('/')[-1])
        
        print('================================================================================================')
        print(f'Retriever: {self.ranker}    Dataset: {self.ds}    Variant Type: {self.method}')
        print('================================================================================================')
        #print('\n')
        #print(' ......... Evaluation with Original QRELS ......... ')
        res_ori = self.evaluate_res_qrels(results, qrels_list_original, version_name, reference_queries)
        #print('\n \n')
        #print(' ......... Evaluation with Generated QRELS ......... ')
        res_gen = self.evaluate_res_qrels(results, qrels_list_generated, version_name, [])

        res_final = pd.merge(res_ori, res_gen, on='Query', how='outer')
        res_final = res_final[['Query','AP(rel=2)_x','AP(rel=2)_y','nDCG@10_x','nDCG@10_y','RBO(p=0.9)_x','BERT_Score']]
        res_final.rename(columns={'AP(rel=2)_x': 'AP(rel=2)_ORI', 'AP(rel=2)_y': 'AP(rel=2)_GEN', 'nDCG@10_x': 'nDCG@10_ORI', 'nDCG@10_y': 'nDCG@10_GEN', 'RBO(p=0.9)_x': 'RBO(p=0.9)'}, inplace=True)
        res_final.sort_values(by='BERT_Score', ascending=False, inplace=True)
        print(res_final)
        out_file_name = f'{self.ranker}.{self.ds}.{self.method}.csv'
        score_out_dir = os.path.join(dirname, f'runs/scores/{out_file_name}')
        res_final.to_csv(score_out_dir,index=None)

if __name__ == "__main__": 
    Evaluation().main()

'''
python -m ROBUST_ranker.evaluate --dataset dl19 --method paraphrase --ranker BM25 --depth 100
python -m ROBUST_ranker.evaluate --dataset dl19 --method generic --ranker BM25 --depth 100
python -m ROBUST_ranker.evaluate --dataset dl19 --method specific --ranker BM25 --depth 100
'''