import torch
import pyterrier as pt
import pandas as pd
from sentence_transformers import CrossEncoder

class CustomCrossEncoder(pt.Transformer):
    def __init__(self, model_name):
        self.cmodel = CrossEncoder(model_name)
    
    def transform(self, df):
        scores = []
        for _, row in df.iterrows():
            score = self.cmodel.predict([(row['query'], row['text'])])
            scores.append(score[0])
        df["score"] = scores  # Replace existing scores
        
        # sort by scores and set final ranks
        run_df = df.copy()
        run_df = run_df.sort_values(["qid", "score"], ascending=[True, False])
        run_df["rank"] = (
            run_df.groupby("qid")["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        return run_df