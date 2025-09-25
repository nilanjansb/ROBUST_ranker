###################################################################################
#                                                                                 #
#                                   CONFIGURATION                                 #
#                                                                                 #
# ##################################################################################

import os

# Current working directory
cwd = os.getcwd()

# index location
#index_path = r'\\DataStore\\index\\'
index_path = r'/DataStore/index/'

# parameters for BM25
bm25_b = 0.75
bm25_k1 = 1.2

# # dataset_configuration WINDOWS
# datasets = {
#     'msmarco_passage':{
#             'name':'msmarco_passage',
#             'irds':'msmarco_passage',
#             'topics': 'train',
#             'index':f'{index_path}\\trecdl_index\\',
#             #'flex': f'{index_path}flex/msmarco.contriever.flex',
#             'flex': f'{index_path}flex\\msmarco.colbertv2.flex',
#             'splade.pisa': f'{index_path}pisa\\msmarco-passage-splade.pisa'
#         },
#     'dl19':{
#             'name':'irds:msmarco-passage/trec-dl-2019/judged',
#             'irds':'msmarco-passage/trec-dl-2019/judged',
#             'topics': 'text',
#             'index':f'{index_path}trecdl_index\\',
#             #'flex': f'{index_path}flex/msmarco.contriever.flex',
#             'flex': f'{index_path}flex\\msmarco.colbertv2.flex',
#             'splade.pisa': f'{index_path}pisa\\msmarco-passage-splade.pisa'
#         },
#     'dl20':{
#             'name':'irds:msmarco-passage/trec-dl-2020/judged',
#             'irds':'msmarco-passage/trec-dl-2020/judged',
#             'topics': 'text',
#             'index':f'{index_path}trecdl_index\\',
#             #'flex': f'{index_path}flex/msmarco.contriever.flex',
#             'flex': f'{index_path}flex\\msmarco.colbertv2.flex',
#             'splade.pisa': f'{index_path}pisa\\msmarco-passage-splade.pisa'
#         },
#     'covid':{
#             'name':'irds:cord19/trec-covid',
#             'irds':'cord19/trec-covid',
#             'topics': 'title',
#             'index':f'{index_path}beir_covid_index\\',
#             'flex': f'{index_path}flex\\covid.contriever.flex'
#         },
#     'touche':{
#             'name':'irds:beir/webis-touche2020',
#             'irds':'beir/webis-touche2020',
#             'topics': 'text',
#             'index':f'{index_path}beir_touche_index\\',
#             'flex': f'{index_path}flex\\touche.colbertv2.flex'
#         },
#     'touchev2':{
#             'name':'irds:beir/webis-touche2020/v2',
#             'irds':'beir/webis-touche2020/v2',
#             'topics': 'text',
#             'index':f'{index_path}beir_touchev2_index\\',
#             'flex': f'{index_path}flex\\touchev2.colbertv2.flex'
#         },
#     'scifact':{
#             'name':'irds:beir/scifact/test',
#             'irds':'beir/scifact/test',
#             'topics': 'text',
#             'index':f'{index_path}beir_scifact_index\\',
#             'flex': f'{index_path}flex\\scifact.contriever.flex'
#         },
#     'fair21':{
#             'name':'irds:trec-fair/2021',
#             'irds':'trec-fair/2021',
#             'topics': 'text',
#             'index':f'{index_path}fair21\\',
#             'flex': f'{index_path}flex\\fair21.new.colbertv2.flex'
#         },
#     'fair21.train':{
#             'name':'irds:trec-fair/2021/train',
#             'irds':'trec-fair/2021/train',
#             'topics': 'text',
#             'index':f'{index_path}fair21\\',
#             'flex': f'{index_path}flex\\fair21.new.colbertv2.flex'
#         },
#     'fair21.eval':{
#             'name':'irds:trec-fair/2021/eval',
#             'irds':'trec-fair/2021/eval',
#             'topics': 'text',
#             'index':f'{index_path}fair21\\',
#             'flex': f'{index_path}flex\\fair21.new.colbertv2.flex'
#         },
#     'fair22':{
#             'name':'irds:trec-fair/2022',
#             'irds':'trec-fair/2022',
#             'topics': 'text',
#             'index':f'{index_path}fair22\\',
#             'flex': f'{index_path}flex\\fair22.new.colbertv2.flex'
#         },
#     'fair22.train':{
#             'name':'irds:trec-fair/2022/train',
#             'irds':'trec-fair/2022/train',
#             'topics': 'text',
#             'index':f'{index_path}fair22\\',
#             'flex': f'{index_path}flex\\fair22.new.colbertv2.flex'
#         },
#     'fair22.eval':{
#             'name':'irds:trec-fair/2022',
#             'irds':'trec-fair/2022',
#             'topics': 'text',
#             'index':f'{index_path}fair22\\',
#             'flex': f'{index_path}flex\\fair22.new.colbertv2.flex'
#         }
# }

# dataset_configuration LINUX
datasets = {
    'msmarco_passage':{
            'name':'msmarco_passage',
            'irds':'msmarco_passage',
            'topics': 'train',
            'index':f'{index_path}/trecdl_index/',
            #'flex': f'{index_path}flex/msmarco.contriever.flex',
            'flex': f'{index_path}flex/msmarco.colbertv2.flex',
            'splade.pisa': f'{index_path}pisa/msmarco-passage-splade.pisa'
        },
    'dl19':{
            'name':'irds:msmarco-passage/trec-dl-2019/judged',
            'irds':'msmarco-passage/trec-dl-2019/judged',
            'topics': 'text',
            'index':f'{index_path}trecdl_index/',
            #'flex': f'{index_path}flex/msmarco.contriever.flex',
            'flex': f'{index_path}flex/msmarco.colbertv2.flex',
            'splade.pisa': f'{index_path}pisa/msmarco-passage-splade.pisa'
        },
    'dl20':{
            'name':'irds:msmarco-passage/trec-dl-2020/judged',
            'irds':'msmarco-passage/trec-dl-2020/judged',
            'topics': 'text',
            'index':f'{index_path}trecdl_index/',
            #'flex': f'{index_path}flex/msmarco.contriever.flex',
            'flex': f'{index_path}flex/msmarco.colbertv2.flex',
            'splade.pisa': f'{index_path}pisa/msmarco-passage-splade.pisa'
        },
    'covid':{
            'name':'irds:cord19/trec-covid',
            'irds':'cord19/trec-covid',
            'topics': 'title',
            'index':f'{index_path}beir_covid_index/',
            'flex': f'{index_path}flex/covid.contriever.flex'
        },
    'touche':{
            'name':'irds:beir/webis-touche2020',
            'irds':'beir/webis-touche2020',
            'topics': 'text',
            'index':f'{index_path}beir_touche_index/',
            'flex': f'{index_path}flex/touche.colbertv2.flex'
        },
    'touchev2':{
            'name':'irds:beir/webis-touche2020/v2',
            'irds':'beir/webis-touche2020/v2',
            'topics': 'text',
            'index':f'{index_path}beir_touchev2_index/',
            'flex': f'{index_path}flex/touchev2.colbertv2.flex'
        },
    'scifact':{
            'name':'irds:beir/scifact/test',
            'irds':'beir/scifact/test',
            'topics': 'text',
            'index':f'{index_path}beir_scifact_index/',
            'flex': f'{index_path}flex/scifact.contriever.flex'
        },
    'fair21':{
            'name':'irds:trec-fair/2021',
            'irds':'trec-fair/2021',
            'topics': 'text',
            'index':f'{index_path}fair21/',
            'flex': f'{index_path}flex/fair21.new.colbertv2.flex'
        },
    'fair21.train':{
            'name':'irds:trec-fair/2021/train',
            'irds':'trec-fair/2021/train',
            'topics': 'text',
            'index':f'{index_path}fair21/',
            'flex': f'{index_path}flex/fair21.new.colbertv2.flex'
        },
    'fair21.eval':{
            'name':'irds:trec-fair/2021/eval',
            'irds':'trec-fair/2021/eval',
            'topics': 'text',
            'index':f'{index_path}fair21/',
            'flex': f'{index_path}flex/fair21.new.colbertv2.flex'
        },
    'fair22':{
            'name':'irds:trec-fair/2022',
            'irds':'trec-fair/2022',
            'topics': 'text',
            'index':f'{index_path}fair22/',
            'flex': f'{index_path}flex/fair22.new.colbertv2.flex'
        },
    'fair22.train':{
            'name':'irds:trec-fair/2022/train',
            'irds':'trec-fair/2022/train',
            'topics': 'text',
            'index':f'{index_path}fair22/',
            'flex': f'{index_path}flex/fair22.new.colbertv2.flex'
        },
    'fair22.eval':{
            'name':'irds:trec-fair/2022',
            'irds':'trec-fair/2022',
            'topics': 'text',
            'index':f'{index_path}fair22/',
            'flex': f'{index_path}flex/fair22.new.colbertv2.flex'
        }
}

rerankers = {
    'zephyr':{
        'model_name':'HuggingFaceH4/zephyr-7b-beta',
        'tokenizer': 'HuggingFaceH4/zephyr-7b-beta'
    },
    'flanxl':{
        'model_name':'google/flan-t5-xl',
        'tokenizer': 'google/flan-t5-xl'
    },
    'rank-zephyr':{
        'model_name':'castorini/rank_zephyr_7b_v1_full',
        'tokenizer': 'castorini/rank_zephyr_7b_v1_full'
    },
    'rank-vicuna':{
        'model_name':'castorini/rank_zephyr_7b_v1_full',
        'tokenizer': 'castorini/rank_zephyr_7b_v1_full'
    }
}
