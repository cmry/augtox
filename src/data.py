from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class TaskLoader(object):

    def __init__(self, ddir='../data/'):
        self.dir = ddir
        self.df = pd.read_csv(ddir + 'full_set.csv', index_col=0)
        self.df_aug = self.merge_augmentations()

    def merge_augmentations(self):
        aug_df = False
        for file_name in glob(self.dir + 'upsample*'):
            cur_df = pd.read_csv(file_name, index_col=0)
            cur_df['augmenter'] = [file_name.split('_')[1]] * len(cur_df)
            cur_df['label'] = [1] * len(cur_df)
            cur_df.rename(columns={'generation': 'text'}, inplace=True)
            if isinstance(aug_df, bool):
                aug_df = cur_df
            else:
                aug_df = aug_df.append(cur_df, ignore_index=True)
        return aug_df

    def mix_augmentations(self, model_name, ix, aug_only=False):
        """Note that this is dependent on model_name, so assumes arch loop."""
        self.df['augmenter'] = ['none'] * len(self.df)
        self.df['original_index'] = self.df.index
        if not aug_only:
            df = self.df[self.df.index.isin(ix)].append(
                self.df_aug[(self.df_aug['augmenter'] == model_name) &
                            (self.df_aug['original_index'].isin(ix))],
                ignore_index=True)
        else:
            df = self.df_aug[(self.df_aug['augmenter'] == model_name) &
                             (self.df_aug['original_index'].isin(ix))]
        df = df.sort_values('original_index')
        return df

    def get_split_ix(self):
        ix_train, ix_val, ix_test = [], [], []
        for _set in self.df['set'].unique():
            subset = self.df[self.df['set'] == _set]
            tr, te, tl, _ = train_test_split(subset.index, subset['label'],
                                             shuffle=False, test_size=0.1)
            tr, tv, _, _ = train_test_split(tr, tl,
                                            shuffle=False, test_size=0.1)
            ix_train += list(tr)
            ix_val += list(tv)
            ix_test += list(te)
        return ix_train, ix_val, ix_test

    def get_fit_pairs(self, df, weight='plain'):
        """Convert to field lists and filter empty entries."""
        X, y = df['text'].tolist(), df['label'].tolist()
        Xf, yf, buffer, prep = [], [], False, ''
        for xi, yi in zip(X, y):
            if xi:
                if yi and not buffer:
                    buffer = True
                elif not yi and buffer:
                    buffer = False
                    prep = ''
                elif yi and buffer:
                    prep = '' if weight == 'plain' else '<A> '
                Xf.append(prep + xi)
                yf.append(yi)
        return shuffle(Xf, yf, random_state=42)  # inner shuffle

    def load(self, task='vanilla', augmenter='bert', weight='plain',
             aug_only=False):
        """Main loader function to pass stuff to train, dev and test."""
        ix_train, ix_val, ix_test = self.get_split_ix()
        df_train = self.df[self.df.index.isin(ix_train)]
        df_val = self.df[self.df.index.isin(ix_val)]
        df_test = self.df[self.df.index.isin(ix_test)]
        if task == 'augment-train':
            df_train = \
                self.mix_augmentations(augmenter, ix_train, aug_only=False)
            df_train = df_train.fillna('')
        elif task == 'augment-test':
            df_test = self.mix_augmentations(augmenter, ix_test)
            df_test = df_test.fillna('')
        return df_train, df_val, df_test
