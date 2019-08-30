import torch
import pytorch_transformers as pt

from lilbert.lilbert import *       # TODO: reset to vanilla weights !!!


def run(lr=.001,
        NUMCLASSES=5,
        LILBERTDIM=420,
        ALPHA=.5,
        BETA=1.):
    # TODO: finish this script

    # 1. TODO: get a BERT, and finetune it on some classification task
    bert, tok = get_bert()
    bertclassifier = BertClassifier(bert, 768, NUMCLASSES)
    m = BertClassifierModel(bertclassifier)
    # TODO: finetune

    # 2. TODO: train a lil bert on same task for reference
    # get a new BERT, make a lil BERT out of it
    lilbert, _ = get_bert()
    lilbert = make_lil_bert(lilbert, LILBERTDIM)
    lilbertclassifier = BertClassifier(lilbert, LILBERTDIM, NUMCLASSES)
    lilm = BertClassifierModel(lilbertclassifier)
    # TODO: train on same task

    # 3. TODO: distilling a lil BERT from a teacher big BERT
    lilbert, _ = get_bert()
    lilbert = make_lil_bert(lilbert, LILBERTDIM)
    lilbertclassifier = BertClassifier(lilbert, LILBERTDIM, NUMCLASSES)
    lilm_distill = BertDistillModel(bertclassifier, lilbertclassifier, alpha=ALPHA)
    # TODO: train on same task

    # 4. TODO: distilling a lil BERT form a teacher big BERT with ATTENTION
    lilbert, _ = get_bert()
    lilbert = make_lil_bert(lilbert, LILBERTDIM)
    lilbertclassifier = BertClassifier(lilbert, LILBERTDIM, NUMCLASSES)
    lilm_distill = BertDistillWithAttentionModel(bertclassifier, lilbertclassifier, alpha=ALPHA, beta=BETA)
    # TODO: train on same task
