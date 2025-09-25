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


class BertScorer():
    def __init__(self, **kwargs) -> None:

        # parse the command line 
        CLI=argparse.ArgumentParser()

        CLI.add_argument("--dataset", type=str)
        #CLI.add_argument("--method", type=str, choices=['paraphrase', 'generic', 'specific'])
        CLI.add_argument("--variants", type=int)
        CLI.add_argument("--ranker", type=str, choices=['BM25','ColBERT','monoT5','SPLADE','SBERT','CE','SparseCE'])
        CLI.add_argument("--depth", type=int, default=100)
        CLI.add_argument("--random_seed", type=int, default=42)
        CLI.add_argument("--verbose", type=bool, default=False)
        CLI.add_argument("--test", type=bool, default=False)
        CLI.add_argument("--model_name", type=str, default='gpt-4o-mini')
        
        args = CLI.parse_args()

        self.ds = args.dataset
        #self.method = args.method
        self.variants = args.variants
        self.ranker = args.ranker
        self.depth = args.depth
        self.random_seed = args.random_seed
        self.verbose = args.verbose
        self.test = args.test
        self.model_name = args.model_name
        self.bertscore = load("bertscore")

    def list_avg(self, lst):
        # try:
        #     lst = lst[1:]
        #     avg = sum(lst)/len(lst)
        # except:
        #     avg=-1
        # 
        avg = sum(lst)/len(lst)
        avg = round(avg,4)
        return avg

    def main(self):

        # import the dataset
        dataset = pt.datasets.get_dataset(configuration.datasets[self.ds]['name'])
        qrels = dataset.get_qrels()
        dirname = os.path.dirname(__file__)
        text_res_dir = os.path.join(dirname, f'runs/text_res/')

        methods = ['paraphrase','generic','specific']
        #methods = ['paraphrase']
        scores_results = pd.DataFrame()

        

        for method in tqdm(methods, total = len(methods)):
            final_files_path = glob.glob(f'{text_res_dir}{self.ranker}.{self.ds}.{method}.*')
            final_files_path.sort()

            print(final_files_path)

            per_query_set_scores = []

            for final_file_path in final_files_path:
                var = final_file_path.split('/')[-1].split('.')[-2]
                text_res_labeled_dir = os.path.join(dirname, f'runs/gen_qrels/')
                text_res = pd.read_csv(final_file_path,sep='\t')
                text_res = RRanker.result_type_sync(self, text_res)
                if var=='0':
                    reference_text = list(text_res['query'].unique())
                else:
                    generated_text = list(text_res['query'].unique())
                    per_query_set_score = self.bertscore.compute(predictions=generated_text, references=reference_text, lang="en")
                    per_query_set_scores.append(per_query_set_score)
            
            f1_scores = []
            std_f1_scores = []
            p_scores = []
            std_p_scores = []
            r_scores = []
            std_r_scores = []

            row_names = []
            ver = 1
            for bscore_f1 in per_query_set_scores:
                f1_score = self.list_avg(bscore_f1['f1'])
                std_f1_score = round(statistics.stdev(bscore_f1['f1']),4)
                p_score = self.list_avg(bscore_f1['precision'])
                std_p_score = round(statistics.stdev(bscore_f1['precision']),4)
                r_score = self.list_avg(bscore_f1['recall'])
                std_r_score = round(statistics.stdev(bscore_f1['recall']),4)

                f1_scores.append(f1_score)
                std_f1_scores.append(std_f1_score)
                p_scores.append(p_score)
                std_p_scores.append(std_p_score)
                r_scores.append(r_score)
                std_r_scores.append(std_r_score)
                row_names.append(f'Version {ver}')
                ver = ver+1

            row_names.append('Mean')
            f1_scores.append(self.list_avg(f1_scores))
            p_scores.append(self.list_avg(p_scores))
            r_scores.append(self.list_avg(r_scores))

            method_vals = [method]*len(f1_scores)

            res_dict = {'Method':method_vals,'Value':row_names,'Precision':p_scores, 'Recall':r_scores, 'F1':f1_scores}
            scores_result = pd.DataFrame(res_dict)
            scores_results = pd.concat([scores_results,scores_result])
        print(scores_results)
if __name__ == "__main__": 
    BertScorer().main()

'''
python -m ROBUST_ranker.bscore --dataset dl19 --method paraphrase --ranker BM25 --depth 100
'''