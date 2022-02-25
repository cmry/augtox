# flake8: noqa:E702
import warnings
import os
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# reproducibility bit ----------------
from random import seed; seed(42)
from numpy.random import seed as np_seed; np_seed(42)
import os; os.environ['PYTHONHASHSEED'] = str(42)
from torch.cuda import manual_seed as cuda_seed; cuda_seed(42)
from torch import manual_seed; manual_seed(42)
# -----------------------------------

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from transformers import logging

from data import TaskLoader
from clfs import BERTClassifier, SVMClassifier


class Logger(object):

    def __init__(self, conf):
        self.dir = Path('../logs/runs.csv')
        self.log = self.load(conf)

    def load(self, cnf):
        if not self.dir.is_file():
            os.makedirs(os.path.dirname(self.dir), exist_ok=True)
            with open(self.dir, 'w') as fo:
                head = ['time'] + list(cnf.keys()) + ['dset', 'model', 'score']
                fo.write(',' + ','.join(head) + '\n')
        return pd.read_csv(self.dir, index_col=0)

    def save(self, exp_vals, dset, model, score):
        time = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
        with open(self.dir, 'w') as fo:
            self.log.loc[len(self.log.index)] = [time] + list(exp_vals) + \
                                                [dset, model, score]
            self.log.to_csv(fo)

    def check(self, conf):
        return ((self.log['iter'] == conf['iter']) &
                (self.log['weight'] == conf['weight']) &
                (self.log['task'] == conf['task']) &
                (self.log['aug'] == conf['aug'])).any()


class Experiment(object):

    def __init__(self, iteration, weight, task, augmenter):
        self.tl = TaskLoader()
        self.conf = {'iter': iteration, 'weight': weight, 'task': task,
                     'aug': augmenter}
        self.logger = Logger(self.conf)
        self.it = iteration

    def fit_clf(self, model, X_train, y_train, X_val, y_val):
        if self.conf['task'] == 'augment-test':  # NOTE: speed-up
            _task, _augmenter = 'vanilla', 'no-aug'
        _task, _augmenter = self.conf['task'], self.conf['aug']
        c = model(task=_task, augmenter=_augmenter, weight=self.conf['weight'],
                  it=self.it)
        c.fit(X_train, y_train, X_val, y_val)
        return c

    def train(self):
        clfs = []
        for model in [SVMClassifier, BERTClassifier]:
            train_df, val_df, test_df = \
                self.tl.load(task=self.conf['task'], augmenter=self.conf['aug'],
                             aug_only=(model == BERTClassifier))
            X_train, y_train = \
                self.tl.get_fit_pairs(train_df, weight=self.conf['weight'])
            X_val, y_val = self.tl.get_fit_pairs(val_df)

            if model == SVMClassifier and self.it:  # needs 1 round
                continue

            clfs.append(self.fit_clf(model, X_train, y_train, X_val, y_val))

        return test_df, clfs

    def test(self, dset, test_df, models):
        if dset != 'all':
            test = test_df[test_df['set'] == dset]
        else:
            test = test_df
        X_test, y_test = self.tl.get_fit_pairs(test)

        for model in models:
            logging.set_verbosity_error()
            ŷ = model.predict(X_test)
            yield model, f1_score(ŷ, y_test, average='binary', pos_label=1)

    def run(self):
        if self.logger.check(self.conf):
            print("Skipping:", '-'.join([str(x) for x in self.conf.values()]))
            return
        else:
            print("Running:", '-'.join([str(x) for x in self.conf.values()]))
        test_df, models = self.train()

        if self.conf['task'] == 'augment-test':
            sets = ['all']
        else:
            sets = ['all', 'asken', 'msp', 'agg', 'xu', 'ytb']
        for dset in sets:
            for model, score in self.test(dset, test_df, models): 
                self.logger.save(self.conf.values(), dset, str(model), score)

if __name__ == '__main__':
    for i in range(5): # NOTE: if robustness checks are required
        seed(42 + i)
        np_seed(42 + i)
        cuda_seed(42 + i)
        manual_seed(42 + i)
        os.environ['PYTHONHASHSEED'] = str(42)
        for weight in ['plain', 'token']:
            if weight == 'token':  # NOTE: can be skipped; no improvement
                continue
            for task in ['vanilla', 'augment-train', 'augment-test']:
                for aug in (['skip', 'eda', 'bert', 'dropout', 'dropout+',
                             'hb', 'hb+', 'bart', 'gpt2', 'gpt3']
                             if task != 'vanilla' else ['no-aug']):
                    Experiment(i, weight, task, aug).run()
