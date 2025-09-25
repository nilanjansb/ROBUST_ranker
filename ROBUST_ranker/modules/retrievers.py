from ROBUST_ranker import configuration
import pyterrier as pt
import warnings
import pandas as pd
from tqdm import tqdm
from pyterrier_t5 import MonoT5ReRanker
import pyterrier_dr
import pyt_splade
import pyterrier_pisa
from pyterrier_pisa import PisaIndex

class Retrievers():
    def remove_symbols(text):
        bad_chars = [';',':','!',"*","/","?","'",'"',"-","_",".","%"]
        for i in bad_chars:
            text = text.replace(i, ' ')
        return str(text)
    
    def clean(queries):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.options.mode.chained_assignment = None
        print('Cleaning queries in progress ...')
        for i in tqdm(range(len(queries))):
            text = queries['query'][i]
            text = text.lower()
            text = Retrievers.remove_symbols(text)
            queries['query'][i] = text
        return queries
    
    def load_model_index(ds,ranker):
        if ranker == 'BM25':
            model = None
            index = pt.IndexFactory.of(configuration.datasets[ds]['index'],memory=True)
        if ranker == 'ColBERT':
            model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
            index = pyterrier_dr.FlexIndex(configuration.datasets[ds]['flex'])
        if ranker == 'monoT5':
            model = MonoT5ReRanker()
            index = pt.IndexFactory.of(configuration.datasets[ds]['index'],memory=True)
        if ranker == 'SPLADE':
            model = pyt_splade.Splade()
            index = PisaIndex(configuration.datasets[ds]['splade.pisa'], stemmer='none')
        return model, index

    def retrieve_bm25(topics, index, depth:int):
        bm25 = pt.BatchRetrieve(index, controls={"wmodel": 'BM25', "bm25.b" : configuration.bm25_b, "bm25.k_1": configuration.bm25_k1})%depth
        pipeline = bm25 >> pt.text.get_text(index, "text") 
        res = pipeline.transform(topics)
        return res
    
    def retrieve_colbert(model, topics, index, depth:int):
        retr_pipeline = model >> index.np_retriever(num_results=depth)
        res = retr_pipeline.transform(topics)
        return res

    def retrieve_monot5(model, topics, index, depth:int):
        bm25 = pt.BatchRetrieve(index, controls={"wmodel": 'BM25', "bm25.b" : configuration.bm25_b, "bm25.k_1": configuration.bm25_k1})%depth
        pipeline = bm25 >> pt.text.get_text(index, "text") >> model
        res = pipeline.transform(topics)
        return res
    
    def retrieve_splade(model, topics, index, depth:int):
        pipeline = model.query_encoder() >> index.quantized()
        res = pipeline.transform(topics)
        return res
    
    def retrieve_doc_text(res, topics, index):
        pipeline =  pt.Transformer.from_df(res) >> pt.text.get_text(index, "text") 
        tres = pipeline.transform(topics)
        return tres
