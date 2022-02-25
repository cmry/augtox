import os
import pickle
from copy import deepcopy
from collections import ChainMap
from glob import glob
from pathlib import Path
from requests.exceptions import HTTPError
from time import sleep

import numpy as np
import torch
from datasets import load_metric
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, make_scorer
# from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, logging)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()


class DSet(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in
                self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BERTClassifier(object):

    def __init__(self, model="bert-base-uncased", task="vanilla",
                 augmenter='', weight='plain', it=0):
        self.tok, self.clf = self.load_model(model)
        self.task = task
        self.aug = augmenter
        self.weight = weight
        self.model = None
        self.it = it
        self.name = f"./../logs/model-bully-{self.it}-{self.weight}-" + \
                    f"{self.task}-{self.aug}"

    def __str__(self):
        return "BERT"

    def load_model(self, model, recur=False):
        try:
            return (AutoTokenizer.from_pretrained(model, use_fast=True),
                    AutoModelForSequenceClassification.from_pretrained(
                        model, num_labels=2))
        except HTTPError:
            recur_add = " still " if recur else " "
            print(f"HuggingFace is{recur_add}down, 😟 going to sleep...")
            sleep(10)
            return self.load_model(model, recur=True)

    def data_wrap(self, X, y):
        toks = self.tok(X, truncation=True, padding='max_length',
                        max_length=128)
        return DSet(toks, y)

    def compute_metrics(self, eval_pred):
        metric = load_metric("f1")
        predictions, labels = eval_pred
        return metric.compute(predictions=np.argmax(predictions, axis=1),
                              references=labels)

    def load(self):
        self.clf = AutoModelForSequenceClassification.\
            from_pretrained(self.get_latest_checkpoint(), num_labels=2)
        return self

    def get_latest_checkpoint(self, base=None):
        check_dir = sorted(
            {int(v.split('/')[-1].replace('checkpoint-', '')):
             v for v in glob((f'{self.name}/checkpoint*') if not base else
             f'./../logs/{self.it}-base/checkpoint*')}.items(),
            reverse=True)
        try:
            return check_dir[0][1]
        except IndexError:
            return

    def fit(self, X, y, X_v, y_v):
        check_dir = self.get_latest_checkpoint(base=True)
        if check_dir:
            self.clf = AutoModelForSequenceClassification.\
                from_pretrained(check_dir, num_labels=2)

        ds_t = self.data_wrap(X, y)
        ds_v = self.data_wrap(X_v, y_v)

        args = TrainingArguments(
            self.name if check_dir else f'./../logs/{self.it}-base',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=10 if not check_dir else 2,
            weight_decay=0.01,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model='f1')
        gym = Trainer(model=self.clf, args=args,
                      train_dataset=ds_t, eval_dataset=ds_v,
                      compute_metrics=self.compute_metrics)

        # NOTE: don't retrain for test augmentation check
        if 'test' not in self.name:
            gym.train()
        self.model = gym
        return gym

    def predict(self, X, y=False):
        if not y:
            y = [0] * len(X)
        if not self.model:
            exit("Pipeline not fitted.")
        X = self.data_wrap(X, y)
        return np.argmax(self.model.predict(X, y).predictions, axis=1)

    def model_predict(self, X, batch_size=50, device='cuda:0'):
        self.clf = self.clf.to(device).eval()
        pred, batch = [], []
        for i, s in enumerate(X):
            batch.append(s)
            if not i % batch_size:
                tokens = self.tok(batch, truncation=True, max_length=128,
                                  padding='max_length', return_tensors='pt'
                                  ).to(device)
                logits = self.clf(**tokens).logits
                pred.append(np.argmax(
                    torch.softmax(logits, dim=1).tolist(), axis=1))
                batch = []
        return list(np.hstack(pred))


class SVMClassifier(object):

    def __init__(self, task='vanilla', augmenter='', weight='plain', it=None):
        """SVM Pipeline object."""
        self.pipeline = {
            ('vect', CountVectorizer(binary=True, ngram_range=(1, 3))): {
                # 'vect__ngram_range': [(1, 1), (1, 2), (1, 3)]
            },
            ('svc', LinearSVC(random_state=42, C=0.01,
                              class_weight='balanced')): {
                # 'svc__C': [1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3],
                # 'svc__loss': ['hinge', 'squared_hinge'],
                # 'svc__class_weight': [None, "balanced"]
            }
        }
        self.model = None
        self.f1 = make_scorer(f1_score, average='binary', pos_label=1)
        self.task = task
        self.aug = augmenter
        self.weight = weight
        self.it = it
        self.dir = Path(f"./../logs/model-bully-{self.it}-{self.weight}-" +
                        f"{self.task}-{self.aug}/svm.pickle")

    def __str__(self):
        return "SVM"

    def _iter(self, X, Xv):
        X_range = list(range(len(X)))
        Xv_range = list(range(X_range[-1] + 1, X_range[-1] + len(Xv) + 1))
        for _ in range(1):
            yield (X_range, Xv_range)

    def _cv_score(self, clf, p_grid, X, y, Xv, yv):
        """Big evaluation function, handles cross-val."""
        # NOTE: best param settings for all experiments are hard-coded
        # grid = GridSearchCV(estimator=clf, param_grid=p_grid,
        #                     cv=self._iter(X, Xv), scoring=self.f1, n_jobs=18)
        # grid.fit(X + Xv, y + yv)
        # clf = grid.best_estimator_
        clf.fit(X, y)
        return clf

    def fit(self, X_train, y_train, X_val, y_val):
        """Merge params and pipe to scoring. Report nested score if needed."""
        pipe = deepcopy(self.pipeline)
        if 'test' in str(self.dir):  # NOTE: don't retrain for test aug
            self.dir = Path(f"./../logs/model-bully-{self.it}-{self.weight}-" +
                            "vanilla-no-aug/svm.pickle")
        if not self.dir.is_file():
            os.makedirs(os.path.dirname(self.dir), exist_ok=True)
            clf = self._cv_score(
                Pipeline(list(pipe.keys())),
                dict(ChainMap(*pipe.values())), X_train, y_train, X_val, y_val)

            self.model = clf
            pickle.dump(self.model, open(self.dir, 'wb'))
        else:
            self.model = clf = pickle.load(open(self.dir, 'rb'))

        return clf

    def predict(self, X, y=False):
        if not self.model:
            exit("Pipeline not fitted.")
        return self.model.predict(X)
