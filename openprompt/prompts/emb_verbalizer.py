import os
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger
from sentence_transformers import util
from tqdm import tqdm

class EmbVerbalizer(Verbalizer):
    r"""
    This is the implementation of knowledeagble verbalizer, which uses external knowledge to expand the set of label words.
    This class inherit the ``ManualVerbalizer`` class.
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`classes`): The classes (or labels) of the current task.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        max_token_split (:obj:`int`, optional):
        verbalizer_lr (:obj:`float`, optional): The learning rate of the verbalizer optimization.
        candidate_frac (:obj:`float`, optional):
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer = None,
                 model: Optional[PreTrainedModel] = None,
                 classes: Sequence[str] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 max_token_split: Optional[int] = -1,
                 verbalizer_lr: Optional[float] = 5e-2,
                 candidate_frac: Optional[float] = 0.5,
                 pred_temp: Optional[float] = 1.0,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 post_log_softmax: Optional[bool] = True,
                 sumprob: Optional[bool] = False,
                 verbose: Optional[bool] = False
                 ):
        super().__init__(tokenizer=tokenizer, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax

        self.sumprob = sumprob
        self.verbose = verbose
        self.max_token_split = max_token_split
        self.verbalizer_lr = verbalizer_lr
        self.candidate_frac = candidate_frac
        self.pred_temp = pred_temp
        self.embeddings = model.get_input_embeddings()

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()
        if self.verbose:
            print('## Labels:\n', self.label_words)

    @staticmethod
    def delete_common_words(d):
        word_count = {}
        for d_perclass in d:
            for w in d_perclass:
                if w not in word_count:
                    word_count[w] = 1
                else:
                    word_count[w] += 1
        for w in word_count:
            if word_count[w] >= 2:
                for d_perclass in d:
                    if w in d_perclass[1:]:
                        findidx = d_perclass[1:].index(w)
                        d_perclass.pop(findidx + 1)
        return d

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ' '.
        """
        new_label_words = []
        for words in label_words:
            new_label_words.append([prefix + word.lstrip(prefix) for word in words])
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more one token.
        """
        all_ids = []
        label_words = []
        for words_per_label in self.label_words:
            ids_per_label = []
            words_keep_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if 0 < self.max_token_split < len(ids):
                    # in knowledgeable verbalizer, the label words may be very rare, so we may want to remove the label words which are not recognized by tokenizer.
                    logger.warning("Word {} is split into {} (>{}) tokens: {}. Ignored.".format(word, len(ids), self.max_token_split, self.tokenizer.convert_ids_to_tokens(ids)))
                    continue
                else:
                    words_keep_per_label.append(word)
                    ids_per_label.append(ids)
            label_words.append(words_keep_per_label)
            all_ids.append(ids_per_label)
        self.label_words = label_words

        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        words_ids_mask = [[[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_per_label] + [[0] * max_len] * (max_num_label_words - len(ids_per_label)) for ids_per_label in all_ids]
        words_ids = [[ids + [0] * (max_len - len(ids)) for ids in ids_per_label] + [[0] * max_len] * (max_num_label_words - len(ids_per_label)) for ids_per_label in all_ids]

        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False)  # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        if self.verbose:
            print('## Weight:', self.label_words_weights)
        print("## Num of label words for each label: {}".format(self.label_words_mask.sum(-1).cpu().tolist()), flush=True)

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:
        (1) Project the logits into logits of label words
        if self.post_log_softmax is True:
            (2) Normalize over all label words
            (3) Calibrate (optional)
        (4) Aggregate (for multiple label words)
        Args:
            logits (:obj:`torch.Tensor`): The original logits.
        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project Output: (batch_size, num_classes) or (batch_size, num_classes, num_label_words_per_label)
        # logits: batch_size * vocab_size

        batch_size = logits.shape[0]
        label_words_weights = F.softmax(self.label_words_weights - 10000 * (1 - self.label_words_mask), dim=-1)

        if self.sumprob:
            probs = logits.softmax(dim=-1)
            label_words_probs = probs[:, self.label_words_ids]
            label_words_probs = self.handle_multi_token(label_words_probs, self.words_ids_mask)
            new_label_words_stat = (label_words_probs * self.label_words_mask * label_words_weights).sum(-1)
        else:
            label_words_logits = logits[:, self.label_words_ids]
            label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
            new_label_words_stat = (label_words_logits * self.label_words_mask * label_words_weights).sum(-1)

        final_stat = torch.zeros(batch_size, self.num_classes)
        cur_sum = 0
        for i in range(self.num_classes):
            cur_label_words_stat = new_label_words_stat[:, cur_sum:cur_sum + self.label_len[i]].max(dim=1).values
            cur_sum += self.label_len[i]
            final_stat[:, i] = cur_label_words_stat.cpu()
        return final_stat

    def from_file(self, select_num: Optional[int] = 40, truncate: Optional[float] = -1., dataset_name: Optional[str] = " ",
                  path: Optional[str] = None, tmodel: Optional[str] = 'roberta'):
        print('Dataset name:', dataset_name)
        if dataset_name == 'agnews':
            class_names = [[" world", " politics"], [" sports"], [" business"], [" technology", ' science']]
        elif dataset_name == 'dbpedia':
            class_names = [[' company'],  # 0
                           [' school'],  # 1
                           [' artist'],  # 2
                           [' sports'],  # 3
                           [' politics', ' office'],  # 4
                           [' transportation', ' car', ' bus', ' train'],  # 5
                           [' building', ' construct', ' room', ' tower'],  # 6
                           [' river', ' lake', ' mountain'],  # 7
                           [' village'],  # 8
                           [' animal', ' pet'],  # 9
                           [' plant'],  # 10
                           [' album'],  # 11
                           [' film'],  # 12
                           [' book', ' publication']  # 13
                           ]
            if tmodel == 'bert':
                class_names[3].extend(['runner', 'gym'])
                class_names[4].extend(['politician', 'official'])
                class_names[9].extend(['insect'])
        elif dataset_name == 'imdb':
            class_names = [[' bad'], [' good']]
        elif dataset_name == 'amazon':
            class_names = [[' bad'], [' good']]
        elif dataset_name == 'sst2':
            class_names = [[' terrible'], [' great']]
        elif 'mnli' in dataset_name:
            class_names = [[' yes'], [' maybe'], [' no']]
        elif dataset_name == 'qnli':
            class_names = [[' Yes', ' Indeed', ' Overall'], [' No', ' Well', ' However']]
        elif dataset_name == 'cola':
            class_names = [[' wrong'], [' true']]
        elif dataset_name == 'mrpc':
            class_names = [[' No'], [' Yes']]
        elif dataset_name == 'qqp':
            class_names = [[' No'], [' Yes']]
        elif dataset_name == 'rte':
            class_names = [[' Yes'], [' No']]
        elif dataset_name == 'stsb':
            class_names = [[' No'], [' Yes']]
        else:
            raise NotImplementedError

        sum_label_len = 0
        all_label_len = []
        all_label_names = []
        for c in class_names:
            sum_label_len += len(c)
            all_label_len.append(len(c))
            all_label_names.extend([i for i in c])

        if tmodel in ['roberta', 'bert', 'electra', 'deberta-v2', 'spanbert']:
            tok_id = 1
        else:
            tok_id = 0

        print('Select num:', select_num, 'Truncate:', truncate, 'Token ID:', tok_id)
        # if os.path.exists(path):
        #     print('Load from:', path)
        #     sim_mat = torch.load(path)
        # else:
        sim_mat = torch.zeros(sum_label_len, self.tokenizer.vocab_size)
        for i in tqdm(range(sim_mat.shape[0]), desc='Preprocessing similarity matrix'):
            label_words_ids = self.tokenizer(all_label_names[i], padding=True, return_tensors="pt").input_ids[:, tok_id]
            base_emb = self.embeddings(label_words_ids)
            for j in range(self.tokenizer.vocab_size):
                temp_emb = self.embeddings(torch.tensor([j]))
                sim_mat[i][j] = util.cos_sim(temp_emb, base_emb)[0][0].cpu()
        # torch.save(sim_mat, f'{dataset_name}_{tmodel}_cos.pt')

        assert truncate >= 0. or select_num >= 0

        label_list = []
        weight_list = []
        for i in range(len(all_label_names)):
            if select_num == -1:
                indices = (sim_mat[i] > truncate).nonzero().flatten()
                values = sim_mat[i][indices]
            else:
                values, indices = torch.topk(sim_mat[i], select_num)
            label_list.append([self.tokenizer.decode([tok]) for tok in indices])
            weight_list.append(values.cpu().tolist())

        max_num_label_words = max(len(label) for label in label_list)
        weight_list = [values + [0.] * (max_num_label_words - len(values)) for values in weight_list]

        self.label_words_weights = nn.Parameter(torch.tensor(weight_list), requires_grad=False)
        self.label_words = label_list
        self.label_len = all_label_len

        return self