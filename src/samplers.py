# flake8: noqa E702
# reproducibility bit ----------------
from random import seed; seed(42)
from numpy.random import seed as np_seed; np_seed(42)
import os; os.environ['PYTHONHASHSEED'] = str(42)
from torch.cuda import manual_seed as cuda_seed; cuda_seed(42)
from torch import manual_seed; manual_seed(42)
# -----------------------------------

import contextlib
from copy import deepcopy as cp
from tqdm import tqdm

import nltk
import openai
import pandas as pd
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer, AutoConfig,
                          BartForConditionalGeneration, BartTokenizer,
                          pipeline, logging)

from utils import pad_tokenize, replace_word
from eda import eda

logging.set_verbosity_error()


class Filter(object):

    def __init__(self):
        """Filter class for list candidates based on textual features."""        
        self.stop_words = set(open('../utils/stop_words.txt').read().split())

    def pos_change(self, text, to_perturb, candidate):
        """Zero out POS tags that don't match original (excluding N/V)."""
        _, ori_pos_list = zip(*nltk.pos_tag(text, tagset='universal'))
        new_text = text[:]
        new_text[to_perturb] = candidate
        _, new_pos_list = zip(*nltk.pos_tag(new_text, tagset='universal'))

        return new_pos_list != ori_pos_list

    def __call__(self, text, to_perturb, candidates):
        """Remove candidates that have undesiriable qualities."""
        text = pad_tokenize(text)
        for cand in candidates:                             # don't consider:
            if len(cand) < 2 or '.' in cand:                # characters
                continue
            if cand == '[UNK]':                             # unk tokens
                continue
            elif text[to_perturb] in cand:                  # plurals
                continue
            elif text[to_perturb][1:] == cand[1:]:          # caps
                continue
            elif '##' in cand:                              # sub words
                continue
            # elif cand in self.stop_words:                   # stop words
            #     continue
            elif cand in text:                              # duplicate words
                continue
            # elif self.pos_change(text, to_perturb, cand):   # mismatching POS
            #     continue
            else:
                yield cand


class BaseSampler(object):

    def __init__(self):
        self.filter = Filter()

    def sample_from(self):
        raise NotImplementedError

    def __call__(self, text, to_perturb, n_samples, dropout=None):
        """Replace tokens at indicies to_perturb using sampler of choice.

        Parameters
        ----------
        text : ``list``
            List of tokens (str) of the original input text (D).
        to_perturb : ``list``
            List of inidices (int) that need to be replaced.
        n_samples : ``int``
            Maximum amount of samples that will be generated.
        dropout: ``float``, optional (default=None)
            Set to None, no dropout is used, else it's p of dropping weights.

        Returns
        -------
        ``list``
            New text with tokens at to_perturb indices replaced.
        """
        position_candidates = []
        text, to_perturb = cp(text), cp(to_perturb) if to_perturb else text
        for perturb_ix in to_perturb:
            cands = self.sample_from(perturb_ix, text, dropout)
            position_candidates.append(
                (perturb_ix, list(self.filter(text, perturb_ix, cands))))
        for i in range(n_samples):
            new_text = text[:]
            for ix, candidates in position_candidates:
                try:
                    new_text = replace_word(new_text, ix, candidates[i])
                except IndexError:
                    pass
            if new_text != text:
                yield new_text


class BertSampler(BaseSampler):

    def __init__(self, bert='bert-large-cased', k_cand=20, bert_rank=True, 
                 device='cuda:0'):
        """Bert sampler for Masked [2] and Dropout [3] replacement & ranking.

        Parameters
        ----------
        bert: ``str``, optional (default='bert-large-cased')
            Which model BERT to use to extract candidates.
        k_cand: ``int``, optional (default=20)
            Maximum amount of candidates that will be returned per target word.
        bert_rank: ``bool``, optional (default=True)
            Use BERT contextual re-ranking yes/no.
        device: ``str``, optional (default='cuda:0')
            Device to put BERT on ('cpu', 'cuda:n').

        Notes
        -----
        These attacks are discussed in [1]: speficially Section 3.2 (Masked
        Substitution, Dropout Substitution). The re-ranking is discussed in
        Section 3.3 (BERT Similarity).

        Functionality was improved with zero-embedding back-off, and bert
        similarity patched in [4].

        References
        ----------
        [1] Emmery, C., Kádár, Á., & Chrupała, G. (2021, April). Adversarial
            Stylometry in the Wild: Transferable Lexical Substitution Attacks
            on Author Profiling. In Proceedings of the 16th Conference of the
            European Chapter of the Association for Computational Linguistics:
            Main Volume (pp. 2388-2402).
        [2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
            2019. BERT: pre-training of deep bidirectional transformers for
            language understanding. In Proceedings of the 2019 Conference of
            the North American Chapter of theAssociation for Computational
            Linguistics: Human Language Technologies, NAACL-HLT 2019,
            Minneapolis, MN, USA, June 2-7, 2019, Volume1 (Long and Short
            Papers), pages 4171–4186. Association for Computational
            Linguistics.
        [3] Wangchunshu Zhou, Tao Ge, Ke Xu, Furu Wei, and Ming Zhou. 2019.
            Bert-based lexical substitution. In Proceedings of the 57th Annual
            Meeting of the Association for Computational Linguistics, pages
            3368–3373.
        [4] Emmery, C., Kádár, Á., Chrupała, G., & Daelemans, W. (2021, 
            August). Augmenting Toxic Content to Improve Cyberbullying
            Classification Robustness.
        """
        self.filter = Filter()
        self.k_cand = k_cand

        self.b = bert
        self.d = device
        cnf = AutoConfig.from_pretrained(self.b, output_hidden_states=True,
                                         output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.b)
        self.model = AutoModelForMaskedLM.from_pretrained(self.b, config=cnf)
        self.model = self.model.to(self.d).eval()
        self.bert_rank = bert_rank

    def _bert_sim(self, sentence, cand, ix):
        """Get BERT sim between sent and cand (Eq 2 in paper)."""
        seq = self.tokenizer.encode(sentence, return_tensors="pt").to(self.d)
        sentence = replace_word(sentence, ix, cand)
        _seq = self.tokenizer.encode(sentence, return_tensors="pt").to(self.d)

        out = self.model.base_model(seq)
        _out = self.model.base_model(_seq)

        # Equation (2) ----------
        w_ik = torch.mean(torch.stack([layer[:, :, :, ix + 1] for
                          layer in out.attentions]), dim=3)
        score_l = []
        for i in range(len(seq)):
            score_l.append(
                torch.mul(w_ik[i], torch.nn.functional.cosine_similarity(
                    torch.cat(out.hidden_states[-5:-1], dim=2)[:, i, :],
                    torch.cat(_out.hidden_states[-5:-1], dim=2)[:, i, :])))

        return torch.stack(score_l).sum()
        # -----------------------

    def _bert_sim_ranking(self, candidates, original_text, ix):
        """Ranks candidates according to BERT similarty scores by Zhou."""
        synonyms = {}
        for token in candidates:
            current_word = self.tokenizer.decode([token])
            synonyms[current_word] = \
                self._bert_sim(original_text, current_word, ix).item()

        # NOTE: can enable > self.sim_threshold after `if x[1]`
        return [x[0] for x in sorted(synonyms.items(), key=lambda x: x[1],
                                     reverse=True) if x[1]]

    def _dropout_embedding(self, sequence, token_idx, dropout_p, ix_map):
        """Apply dropout to the embedding of the target token."""
        emb = self.model.base_model.embeddings(sequence)
        if token_idx > len(ix_map) - 1:
            return False
        elif isinstance(ix_map[token_idx], tuple):
            mask_start = ix_map[token_idx][0] + 1  # 1 offsets the <bos> token
            mask_end = ix_map[token_idx][-1] + 1
            emb[:, mask_start] = emb[:, mask_start:mask_end, :].\
                                fill_(0).mean(axis=1)  # when oov, fill with 0
            return torch.cat((emb[:, :mask_start + 1], emb[:, mask_end + 1:]),
                            axis=1)
        else:
            emb[:, ix_map[token_idx] + 1] = \
                torch.nn.Dropout(p=dropout_p)(emb[:, ix_map[token_idx] + 1])
            return emb

    def _map_subword_indices(self, text):
        """Produce a list of indices where the subword indices are tuples."""
        bert_tokens = self.tokenizer.tokenize(text)
        indices, buffer, count = [], 0, 0
        for i, token in enumerate(bert_tokens):
            if token.startswith('#') and not buffer and not i:
                indices.append((i, ))
                buffer = i
                count += 1
                continue
            elif token.startswith('#') and not buffer and i:
                indices.pop(-1)
                buffer = i - 1
                indices.append((buffer, i))
                count += 1
                continue
            elif token.startswith('#') and buffer:
                indices[-1] += (i, )
            elif not token.startswith('#') and buffer:
                buffer = 0
            if not buffer:
                indices.append(i)
        return indices

    def _ix_offset(self, text, mask_idx):
        """Offset mask index by bos and amount of subwords left of mask."""
        indices, offset = self._map_subword_indices(text), 0
        for x in indices[:mask_idx]:
            if isinstance(x, tuple):
                offset += len(x) - 1  # -1 for starting index collapsed word
        return indices, offset + 1  # +1 for BOS padding (offset only)

    def _flat_encode(self, text, mask=False):
        """Syntactic sugar to encode a 1-dim instance."""
        if not isinstance(mask, bool):
            text = replace_word(text, mask, self.tokenizer.mask_token)
        return self.tokenizer.encode(text, return_tensors="pt").to(self.d)

    def sample_from(self, to_perturb, text, dropout=None):
        """Given text, propose perturbations according to BERT models.

        Parameters
        ----------
        to_perturb: ``int``, required
            Index of (int) target word that should be attacked (t).
        text: ``list``, required
            List of (str) tokens that should be attacked (D).
        dropout: ``float``, optional (default=0.3)
            Set to None, no dropout is used, else it's p of dropping weights.

        Returns
        -------
        to_perturb: ``list``
            List of (str) synonyms C_t for target word t.
        """
        seq = self._flat_encode(text, mask=False if dropout else to_perturb)
        with torch.no_grad():
            if dropout:
                ix_map, offset = self._ix_offset(text, to_perturb)
                emb = self._dropout_embedding(seq, to_perturb, dropout, ix_map)
                # NOTE: this captures rare tokenizer mismatch bugs
                if isinstance(emb, bool):
                    return []
                token_logits = self.model(inputs_embeds=emb)[0]
                logits = token_logits[:, to_perturb + offset, :]
            else:
                mask_ix = torch.where(seq == self.tokenizer.mask_token_id)[1]
                logits = self.model(seq)[0][0, mask_ix, :]

        top_cands = torch.topk(logits, self.k_cand, dim=1).indices[0].tolist()
        if self.bert_rank:  # rank candidates according to bert_sim
            return self._bert_sim_ranking(top_cands, text, to_perturb)
        else:  # straight decode
            return [self.tokenizer.decode([x]) for x in top_cands]


class BartSampler(BaseSampler):

    def __init__(self, k_cand=20, device='cuda:0'):
        """Uses BART to generate in-line replacements (multiple masks allowed).

        Parameters
        ----------
        device : ``str``, optional
            device ID to put BART on, by default 'cuda:0'

        Notes
        -----
        Some caveats:
        - No Dropout possible (passing embeddings not supported).
        - Might introduce more words than the number of masks.
        - Passing punctuation messes up the input sequence. Don't.
        """
        self.filter = Filter()
        self.k_cand = k_cand
        
        self.d = device
        self.tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained(
            "facebook/bart-large",
            forced_bos_token_id=self.tokenizer.bos_token)
        self.model.to(self.d).eval()

    def sample_from(self, to_perturb, text, *args):
        """Replace tokens at indicies to_perturb using BART replacement.

        Parameters
        ----------
        text : ``list``
            List of tokens (str) of the original input text (D).
        to_perturb : ``list``, optional
            List of tokens (str) that need to be replaced, by default None.
            If multiple indices are given in a list, the generation is _not_
            conditional upon the earlier replacements.

        Returns
        -------
        ``list``
            New text with tokens at to_perturb indices replaced.
        """
        masked_text = replace_word(text, to_perturb, '<mask>')
        seq = self.tokenizer(masked_text, return_tensors='pt').to(self.d)
        probs = self.model(seq.input_ids).logits[0, torch.nonzero(
                    seq.input_ids[0] == self.tokenizer.mask_token_id
                ).item()].softmax(dim=0)
        return self.tokenizer.decode(probs.topk(self.k_cand).indices).split()


class EDASampler():

    def __init__(self):
        """Uses EDA to augment the sentence."""
        pass

    def __call__(self, text, ix, n_samples, dropout=None):
        return eda(text, num_aug=n_samples)


class DebugSampler():

    def __init__(self):
        """Does nothing."""
        pass

    def __call__(self, text, ix, n_samples, dropout=None):
        return [text]


class GPT2Sampler():

    def __init__(self, device='cuda:0', model='gpt2-large'):
        """Uses GPT2 to generate full text given a context.

        Parameters
        ----------
        device : ``str``, optional
            device ID to put GPT2 on, by default 'cuda:0'
        model : ``str``, optional
            change GPT model

        Notes
        -----
        """
        self.d = device
        self.generator = pipeline('text-generation', model=model, device=0)

    def __call__(self, text, ix, n_samples, dropout=None):
        return [self.generator(
                text,
                do_sample=True,
                max_length=70
            )[0]['generated_text'].replace(text, '').replace('\n', ' ')
            for _ in range(n_samples)]


class GPT3Sampler(object):

    def __init__(self, api=False):
        if api:
            self.gpt3 = self.connect_api()

    def connect_api(self):
        openai.organization, openai.api_key = \
            open('../utils/key.txt').read().split(',')
        return openai

    def generate(self, prompt):
        return [x['text'] for x in openai.Completion.create(
            engine="curie",
            prompt=prompt,
            n=7,
            max_tokens=70,
            temperature=0.7,
            frequency_penalty=1.0
        )["choices"]]

    def test_generate(self, text):
        from random import choice
        return [text + " " + choice(["spam", "ham", "eggs", "foo", "bar"])
                for _ in range(5)]

    def get_candidates(self, data):
        emergency_out = open('../data/emergency_set.csv', 'a')
        candidates = {}
        for ix, text in tqdm(data):
            candidates[ix] = []
            try:
                generations = self.generate(text)
                for gen in generations:
                    gen = gen.replace('\n', ' ')
                    candidates[ix].append(gen)
                    emergency_out.write(f"{ix},{gen}\n")
            except Exception as e:
                print("Ran into an error:", e)
                print("... aborting ...")
                break
        emergency_out.close()
        return candidates

    def __call__(self, text, ix, n_samples, dropout=None):
        df = pd.read_csv('../data/gpt3_set.csv', header=0)
        _df = df[df['original_index'] == ix].dropna()
        return list(_df['generation'])[:n_samples]


if __name__ == '__main__':
    print("\n^^^ Ignore this transformers crap\n\n")

    rep = BertSampler()
    text = "You are a retarded dweeb and stupid af. Go fuck yourself."
    to_perturb = [3, 4, 6, 10]

    print(">", text)
    print('-', '\n- '.join([o for o in rep(text, to_perturb, 5, dropout=0.2)]))


    print()
    rep = BartSampler()
    text = "I don't understand why people like u."
    to_perturb = [6, 7]

    print(">", text)
    print('-', '\n- '.join(
        [o for o in rep(text, to_perturb, 5, dropout=None)]))


    print()
    rep = GPT3Sampler()
    to_perturb = 14

    print(">", text)
    print('-', '\n- '.join(
        [o for o in rep(text, to_perturb, 5, dropout=None)]))



    rep = None  # NOTE: clear out GPU memory