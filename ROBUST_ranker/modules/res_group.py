import re
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path
from typing import Literal
import pyterrier as pt
import ir_datasets
import nltk
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from ROBUST_ranker import configuration
from pyndeval import SubtopicQrel, ScoredDoc
import pyndeval

def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def load_run(run_path: Path, ds:str, depth: int) -> pd.DataFrame:
    run = pd.read_csv(
        run_path,
        sep=r"\s+",
        names=["query_id", "Q0", "doc_id", "rank", "score", "run_id"],
        dtype={"query_id": str, "doc_id": str},
    ).drop(columns=["Q0", "run_id"])
    run = run.loc[run["rank"] <= depth]

    dataset = ir_datasets.load(configuration.datasets[ds]['irds'])
    qrels = pd.DataFrame(dataset.qrels_iter())
    docs_store = dataset.docs_store()
    queries = pd.DataFrame(dataset.queries_iter()).rename(columns={"text": "query"})
    run = run.merge(qrels, on=["query_id", "doc_id"], how="left")
    run = run.merge(queries, on="query_id", how="left")
    run["text"] = run["doc_id"].map(lambda x: docs_store.get(x).default_text())
    return run

def load_run_from_res(res, ds:str, depth: int) -> pd.DataFrame:
    run = res.loc[res["rank"] <= depth]
    run = run.rename(columns={'qid': 'query_id', 'docid': 'doc_id'})
    run['doc_id'] = run['doc_id'].astype(object)

    dataset = ir_datasets.load(configuration.datasets[ds]['irds'])
    qrels = pd.DataFrame(dataset.qrels_iter())
    docs_store = dataset.docs_store()
    queries = pd.DataFrame(dataset.queries_iter()).rename(columns={"text": "query"})
    run = run.merge(qrels, on=["query_id", "doc_id"], how="left")
    run = run.merge(queries, on="query_id", how="left")
    #run["text"] = run["doc_id"].map(lambda x: docs_store.get(x).default_text())
    return run


def load_qrels(qrel_id: str) -> pd.DataFrame:
    dataset = ir_datasets.load(qrel_id)
    qrels = pd.DataFrame(dataset.qrels_iter())
    qrels["text"] = qrels["doc_id"].map(
        lambda x: dataset.docs_store().get(x).default_text()
    )
    return qrels


def cluster_docs(
    df: pd.DataFrame, threshold: float, distance: Literal["jaccard", "edit"]
) -> pd.DataFrame:
    clusters = np.full((df.shape[0],), fill_value=-1, dtype=np.int32)
    #for _, query_df in tqdm(df.groupby("query_id")):
    for _, query_df in df.groupby("query_id"):
        if query_df.shape[0] > 1_000:
            query_df = query_df.loc[query_df.loc[:, "relevance"] > 0]
        if query_df.shape[0] > 5_000:
            continue
        idcs = query_df.index.values
        words = query_df["text"].map(nltk.word_tokenize).str[:512].values
        idx_a, idx_b = np.triu(np.ones((len(words), len(words))), k=1).nonzero()
        pairwise_distances = np.full((len(words), len(words)), 10e6)
        if distance == "jaccard":
            distances = np.array(
                [
                    [
                        nltk.jaccard_distance(set(a), set(b))
                        for a, b in zip(words[idx_a], words[idx_b])
                    ]
                ]
            )
        elif distance == "edit":
            distances = np.array(
                [nltk.edit_distance(a, b) for a, b in zip(words[idx_a], words[idx_b])]
            )
        else:
            raise ValueError(f"Unknown distance: {distance}")
        pairwise_distances[idx_a, idx_b] = distances
        pairwise_distances[idx_b, idx_a] = distances
        ac = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=threshold,
        )
        g = ac.fit_predict(pairwise_distances)
        clusters[idcs] = g
    df["iteration"] = clusters
    return df

def make_qrels(grouped_base, ds):
    dataset = pt.datasets.get_dataset(configuration.datasets[ds]['name'])
    qrels = dataset.get_qrels()

    rel_baseline_data = grouped_base.copy()
    rel_qrels_subtopics = []

    for idx,row in rel_baseline_data.iterrows():
        qid = str(row['query_id'])
        did = str(row['doc_id'])
        cid = str(row['iteration'])

        sub_qrels = qrels.loc[qrels['qid']==qid]
        try:
            rel_val = sub_qrels['label'].loc[qrels['docno']==did].values[0]
        except:
            rel_val = 0
        
        if rel_val>=2: # rel-values for MS-MARCO is 2 and 3
            data = SubtopicQrel(qid, cid, did, rel_val)
            rel_qrels_subtopics.append(data)

    return rel_qrels_subtopics

def scoring(rel_novelty_data, rel_qrels_subtopics, measures):
    novelty_data = rel_novelty_data.copy()
    run_tuples = []

    for idx,row in novelty_data.iterrows():
        qid = str(row['query_id'])
        did = str(row['doc_id'])
        score = row['score']
        data = ScoredDoc(qid, did, score)
        run_tuples.append(data)
    
    per_query_eval = pyndeval.ndeval(rel_qrels_subtopics, run_tuples, measures = measures)

    return per_query_eval

def calculate_groups(run_path, depth, dataset, threshold: float = 0.5, distance:str='jaccard'):
    # if type(run_path) == str:
    #     #print('Loading RUN.')
    #     run = load_run(run_path, dataset, depth)
    # else:
    #     #print('Loading RES.')
    run = load_run_from_res(run_path, dataset, depth)
    #print('Clustering.')
    run = cluster_docs(run, threshold, distance)
    run["relevance"] = 1 + run["rank"].max() - run["rank"]

    return run