import pickle
from glob import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from bert_score import BERTScorer
from tqdm import tqdm
from transformers import logging

from clfs import BERTClassifier
from data import TaskLoader
from scoring import meteor_score

logging.set_verbosity_error()
tf.compat.v1.logging.set_verbosity(0)


class Scorer(object):

    def __init__(self):
        self.bert_score = BERTScorer(lang="en", rescale_with_baseline=True)
        self.base_model = pickle.load(open('../logs/model-bully-0-plain-' +
                                           'vanilla-no-aug/svm.pickle', 'rb'))
        self.bert_model = BERTClassifier(augmenter='skip',
                                         task="augment-train").load()

    def fp_rate(self, ref, gen):
        ref_acc = sum(ref) / len(ref)
        gen_acc = sum(gen) / len(gen)
        return ref_acc, ref_acc - gen_acc

    def clfscore(self, R_data, G_data):
        rp = self.base_model.predict(R_data)
        gp = self.base_model.predict(G_data)
        return self.fp_rate(rp, gp)

    def tfscore(self, R_data, G_data, bert=None):
        if not bert:
            bert = self.bert_model
        rp = bert.model_predict(R_data, batch_size=50)
        gp = bert.model_predict(G_data, batch_size=50)
        return self.fp_rate(rp, gp)

    def reference_score(self, R_data, bert):
        ŷ_pln = self.bert_model.model_predict(R_data, batch_size=45)
        ŷ_aug = bert.model_predict(R_data, batch_size=45)
        return self.fp_rate(ŷ_pln, ŷ_aug)

    def bertscore(self, R_data, G_data, avg=True):
        """Measure BERTScore."""
        score = self.bert_score.score(R_data, G_data)[2]
        if avg:
            return round(score.mean().item(), 3)
        else:
            return score

    def bleurtscore(self, R_data, G_data, avg=True):
        """Measure BLEURT."""
        batch, scores = [], []
        for i, (a_doc, x_doc) in tqdm(enumerate(zip(R_data, G_data), start=1)):
            batch.append((a_doc, x_doc))
            if not i % 64:
                R_data, G_data = list(zip(*batch))
                batch_score = self.bleurt.score(
                    references=R_data, candidates=G_data, batch_size=64)
                scores.extend(batch_score)
                batch = []

        if avg:
            return round(np.mean(scores), 3)
        else:
            return scores

    def meteorscore(self, R_data, G_data, avg=True):
        """Measure METEOR."""
        scores = []
        for a_doc, x_doc in zip(R_data, G_data):
            scores.append(meteor_score([a_doc], x_doc))
        if avg:
            return round(np.mean(scores), 3)
        else:
            return scores


def score_data():
    sc = Scorer()
    df_ref = pd.read_csv('../data/positive_set.csv', index_col=0)
    df_sco = pd.DataFrame()
    for _file in glob('../data/upsample_*_set.csv'):
        model = _file.split('_')[1]
        df = pd.read_csv(_file, index_col=0)
        print(df.head())
        gens = [str(x) for x in df['generation'].to_list()]
        refs = [str(x) for x in
                df_ref.loc[df['original_index']]['text'].to_list()]
        svm_org, svm_chg = sc.clfscore(refs, gens)
        bert_org, bert_chg = sc.tfscore(refs, gens)
        df_sco = df_sco.append(pd.DataFrame({
            '_index': df['original_index'].to_list(),
            'reference': refs,
            'candidate': gens,
            'augmenter': model,
            'meteor': sc.meteorscore(refs, gens, avg=False),
            'bertsc': sc.bertscore(refs, gens, avg=False),
            'svm_org': svm_org,
            'svm_chg': svm_chg,
            'bert_org': bert_org,
            'bert_chg': bert_chg
        }))
    df_sco.to_csv('../data/auto_scores.csv')


def eval_aug():
    sc, tl = Scorer(), TaskLoader()
    df_perf = pd.DataFrame()
    models = [model_dir.split('-')[-1] for model_dir
              in glob('../logs/*-*-plain-augment-train*')
              if 'gpt3' not in model_dir]
    for i, augmented_model in tqdm(enumerate(models)):
        for j, augmented_classifer in tqdm(enumerate(models)):
            _, _, test_df = tl.load(task='augment-test', weight='plain',
                                    augmenter=augmented_model, aug_only=True)
            test_set = test_df[test_df['label'] == 1]['text']
            clf = BERTClassifier(augmenter=augmented_classifer,
                                 task="augment-train").load()
            bert_acc, bert_chg = sc.reference_score(test_set, clf)
            df_perf = df_perf.append(pd.DataFrame({
                'augmented_by': augmented_model,
                'classified_by': augmented_classifer,
                'classification_accuracy': bert_acc,
                'classification_change': bert_chg
            }, index=[i + j]))
    df_perf.to_csv('../data/bert_scores.csv')


if __name__ == '__main__':
    # score_data()
    eval_aug()
