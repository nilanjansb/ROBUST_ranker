# ROBUST_Ranker

**A robust evaluation framework for TREC Deep Learning tracks (2019 \& 2020) using BM25, monoT5, ColBERT, and SPLADE rankers.**

***

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Virtual Environment](#virtual-environment)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

***

## Overview

This repository implements robust ranking experiments on the TREC Deep Learning 2019 and 2020 datasets. It supports three query-variant methods (`paraphrase`, `generic`, `specific`) and four rankers (`BM25`, `monoT5`, `ColBERT`, `SPLADE`) to:

- Create ranked result sets.
- Generate relevance labels (QRELS).
- Compute evaluation metrics (e.g., nDCG).

***

## Features

- **Variant-based ranking**: Explore multiple query variations per original query.
- **Multi-ranker support**: Integrate sparse (BM25, SPLADE) and dense (monoT5, ColBERT) rankers.
- **Configurable depth \& few-shot**: Control ranking depth and few-shot labeling.
- **Automated pipelines**: Single commands for end-to-end experiments.

***

## Installation

```bash
git clone https://github.com/your-username/ROBUST_Ranker.git
cd ROBUST_Ranker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


***

## Virtual Environment

| VENV Name | Description |
| :-- | :-- |
| `main` | Core ranking \& evaluation |
| `GenRelLabel` | Generate relevance labels (QRELS) |
| `Evaluate` | Evaluation metrics computation |

Activate with:

```bash
source venv/bin/activate
```


***

## Usage

Run experiments via the `ROBUST_ranker` module. Replace placeholders as needed:

```bash
# 1. Create rankings
python -m ROBUST_ranker.main \
  --dataset {dl19|dl20} \
  --method {paraphrase|generic|specific} \
  --variants <N> \
  --ranker {BM25|monoT5|ColBERT|SPLADE} \
  --depth <D>

# 2. Generate QRELS
python -m ROBUST_ranker.generate_relevance_labels \
  --dataset {dl19|dl20} \
  --method {paraphrase|generic|specific} \
  --ranker {BM25|monoT5|ColBERT|SPLADE} \
  --few_shot_count <K> \
  --depth <D>

# 3. Evaluate results
python -m ROBUST_ranker.evaluate \
  --dataset {dl19|dl20} \
  --method {paraphrase|generic|specific} \
  --ranker {BM25|monoT5|ColBERT|SPLADE} \
  --depth <D>
```

**Examples:**

```bash
python -m ROBUST_ranker.main --dataset dl20 --method paraphrase --variants 5 --ranker BM25 --depth 100
python -m ROBUST_ranker.generate_relevance_labels --dataset dl20 --method generic --ranker ColBERT --few_shot_count 0 --depth 100
python -m ROBUST_ranker.evaluate --dataset dl20 --method specific --ranker SPLADE --depth 10
```


***

## Configuration

All parameters can be customized:

- `--dataset`: `dl20` or `dl19`
- `--method`: `paraphrase`, `generic`, `specific`
- `--variants`: Number of query variants (e.g., `3`, `5`, `15`)
- `--ranker`: `BM25`, `monoT5`, `ColBERT`, `SPLADE`
- `--depth`: Retrieval depth (e.g., `10`, `100`)
- `--few_shot_count`: Number of few-shot examples for QREL generation

***
