from glob import glob

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from transformers import pipeline

from samplers import (BartSampler, BertSampler, EDASampler, GPT2Sampler,
                      GPT3Sampler, DebugSampler)
from utils import pad_tokenize


class DataHandler(object):

    def __init__(self, set_names=None):
        """Handles and aggregates the data.

        Parameters
        ----------
        set_names : list, optional
            List of strings containing alternative set names, by default None
        """
        self.dir = '../amica/corpora/'
        self.set_names = ['msp_set', 'xu_set', 'agg_set', 'asken_set',
                          'ytb_set'] if not set_names else set_names
        try:
            open('../data/full_set.csv', 'r')
        except FileNotFoundError:
            self.init()

    def init(self):
        """Initialize by filterering data and writing the aggregates."""
        ddirs = ([x for x in glob(self.dir + '**/**/*.csv') if
                  'up' not in x and any([k in x for k in self.set_names])])
        self.write_sets(self.aggregate_csvs(ddirs))

    def write_sets(self, df):
        """Write the aggregate df to separate files based on filtering."""
        df['text'] = df['text'].replace('', np.nan)
        df = df.dropna()
        df.to_csv('../data/full_set.csv')
        df_pos = pd.DataFrame(df[df['label'] == 1],
                              columns=['label', 'text', 'set'])
        df_pos.to_csv('../data/positive_set.csv')
        df_pos['text'] = df_pos['text'].apply(
            lambda x: ' '.join(x.split(' ')[:30]))
        df_pos.to_csv('../data/short_set.csv')

    def clean_set(self, df):
        """Cleaning for GPT3. Normalizes handles and removes short docs."""
        df['text'] = df['text'].str.replace(r'@[\w]+ ', '@user ', regex=True)
        df['text'] = df['text'][df['text'].str.count(' ') > 3]
        return df

    def aggregate_csvs(self, ddirs):
        """Loads all the sets and aggregates them to a single DataFrame."""
        df = pd.DataFrame()
        for file_name in ddirs:
            set_name = file_name.split('/')[-1].replace('_set.csv', '')
            _df = pd.read_csv(file_name, names=['label', 'text'])
            _df['set'] = [set_name] * len(_df)
            try:
                df = df.append(_df, ignore_index=True)
            except Exception:
                df = _df
        return self.clean_set(df)

    def iter(self):
        """Iterate through the positive instance sets."""
        df = pd.read_csv('../data/short_set.csv', index_col=0)
        return list(zip(tuple(df.index), tuple(df['text'])))

    def store_candidates(self, candidates, model_name='test', aug_write=False):
        """Write upsampled candidates to file suffixed with model_name."""
        if aug_write:
            pd.DataFrame([[ix, candidate] for ix, candidate in candidates],
                         columns=['original_index', 'generation']
                         ).to_csv(f'../data/upsample_{model_name}_set.csv')
            return
        pd.DataFrame([[ix, candidate] for ix, candidates in candidates.items()
                      for candidate in candidates],
                     columns=['original_index', 'generation']
                     ).to_csv(f'../data/upsample_{model_name}_set.csv')


class ClassWeighter(object):

    def __init__(self, min_p=0.005, model='', dev=''):
        """Base steps for loading."""
        self.min_p = min_p
        self.stop_words = set(open('../utils/stop_words.txt').read().split())
        self.model = self.get_model(model, dev)

    def __call__(self, tokens):
        """Simple call wrapper."""
        return self.get_ommission_score(tokens)

    def get_model(self, model, dev):
        """Placeholder for model loader."""
        return NotImplementedError

    def filter_candidates(self, ommission_scores):
        """Filter candidates based on omission score and heurisitcs."""
        candidates, ix = [], 0
        for i, (word, prob) in enumerate(ommission_scores):
            if word in self.stop_words:
                ix += 1
                continue
            if not i - ix:
                pass
            elif prob < self.min_p:
                continue
            candidates.append(word)
        return candidates

    def get_proba_diffs(self, tokens, masks):
        """Get probability differences between original sents and masked."""
        return NotImplementedError

    def get_ommission_score(self, tokens):
        """Delete each position and see how it changes probabilities."""
        masks = [' '.join(tokens[0:i] + tokens[i+1:len(tokens)])
                 for i in range(len(tokens))]
        sent = [' '.join(tokens)]
        yp, _yp = self.get_proba_diffs(sent, masks)
        y, _y = np.array([yp] * len(tokens)), np.array(_yp)
        return list(self.filter_candidates(
                    sorted(list(zip(range(len(tokens)), -(_y-y))),
                           key=lambda x: x[1], reverse=True)))


class SimpleClassWeighter(ClassWeighter):

    def get_model(self, model='', dev=''):
        """Fit NB pipeline to Formspring data."""
        data_file = '../../amica/corpora/kontostathis/formspring/form_set.csv'
        data = open(data_file).read().split('\n')
        clf = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                      use_idf=1, smooth_idf=1,
                                      sublinear_tf=1)),
            ('nb', MultinomialNB())
        ])
        y, X = zip(*[(d[0], d[2:]) for d in data if d])
        clf.fit(X, y)
        return clf

    def get_proba_diffs(self, sent, masks):
        """Use predict proba to get diff between sents and masks."""
        return self.model.predict_proba(sent)[0][1], \
            [x[1] for x in self.model.predict_proba(masks)]


class BertClassWeighter(ClassWeighter):

    def get_model(self, bert, dev):
        return pipeline("text-classification",
                        model=bert, device=int(dev[-1]))

    def get_proba_diffs(self, sent, masks):
        """Use pipeline to get diff between sent and masks."""
        return self.model([sent])[0]['score'], \
            [x['score'] for x in self.model(masks)]


class Augmenter(object):

    def __init__(self, weighter=SimpleClassWeighter, model='', device=''):
        self.weighter = weighter(model=model, dev=device)
        self.handler = DataHandler()

    def augment_words(self, data, sampler, dropout=None):
        """Per-word augmentation with perturbation samplers."""
        for ix, text in tqdm(data):
            tokens = pad_tokenize(text)
            to_perturb = self.weighter(tokens)
            for sample in sampler(text, to_perturb, 5, dropout):
                yield ix, sample

    def augment_text(self, data, sampler):
        """Text-level augmentation with GPT models generating new contexts."""
        for ix, text in tqdm(data):
            for sample in sampler(text, ix, 5):
                yield ix, sample

    def sampler_calls(self, data, sampler_name='dropout'):
        """Wrapper to route data to samplers."""
        if sampler_name == 'dropout':
            return self.augment_words(data, BertSampler(), dropout=0.2)
        if sampler_name == 'bert':
            return self.augment_words(data, BertSampler(), dropout=None)
        if sampler_name == 'bart':
            return self.augment_words(data, BartSampler())
        if sampler_name == 'gpt2':
            return self.augment_text(data, GPT2Sampler())
        if sampler_name == 'gpt3':
            return self.augment_text(data, GPT3Sampler())

    def write_samples(self):
        """Write augmented positive samples to respective dirs."""
        self.handler.store_candidates(
            self.sampler_calls(self.handler.iter(), 'dropout'),
            model_name='dropout', aug_write=True)
        self.handler.store_candidates(
            self.sampler_calls(self.handler.iter(), 'bert'),
            model_name='bert', aug_write=True)
        self.handler.store_candidates(
            self.sampler_calls(self.handler.iter(), 'bart'),
            model_name='bart', aug_write=True)
        self.handler.store_candidates(
            self.sampler_calls(self.handler.iter(), 'gpt2'),
            model_name='gpt2', aug_write=True)


if __name__ == '__main__':
    aug = Augmenter()
    aug.write_samples()
    aug.handler.store_candidates(
        aug.augment_words(aug.handler.iter(), DebugSampler()),
        model_name='skip', aug_write=True
    )
    aug.handler.store_candidates(
        aug.augment_words(aug.handler.iter(), EDASampler()),
        model_name='eda', aug_write=True
    )
    aug.handler.store_candidates(
        aug.augment_words(aug.handler.iter(),
                          BertSampler(bert='GroNLP/hateBERT'), dropout=0.2),
        model_name='hb', aug_write=True
    )

    aug = Augmenter(weighter=BertClassWeighter, model='unitary/toxic-bert',
                    device='cuda:0')
    aug.handler.store_candidates(
        aug.augment_words(aug.handler.iter(), BertSampler(), dropout=0.2),
        model_name='dropout+', aug_write=True
    )
    aug.handler.store_candidates(
        aug.augment_words(aug.handler.iter(),
                          BertSampler(bert='GroNLP/hateBERT'), dropout=0.2),
        model_name='hb+', aug_write=True
    )
