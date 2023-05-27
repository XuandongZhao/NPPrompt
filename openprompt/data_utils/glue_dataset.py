# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from datasets import load_dataset

import pandas as pd
from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


class SST2Processor(DataProcessor):
    """
    `SST-2 <https://nlp.stanford.edu/sentiment/index.html>`_ dataset is a dataset for sentiment analysis. It is a modified version containing only binary labels (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) on top of the original 5-labeled dataset released first in `Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank <https://aclanthology.org/D13-1170.pdf>`_

    We use the data released in `Making Pre-trained Language Models Better Few-shot Learners (Gao et al. 2020) <https://arxiv.org/pdf/2012.15723.pdf>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.lmbff_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SST-2"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        dev_dataset = processor.get_dev_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 2
        assert processor.get_labels() == ['0','1']
        assert len(train_dataset) == 6920
        assert len(dev_dataset) == 872
        assert len(test_dataset) == 1821
        assert train_dataset[0].text_a == 'a stirring , funny and finally transporting re-imagining of beauty and the beast and 1930s horror films'
        assert train_dataset[0].label == 1

    """

    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
        self.dataset = load_dataset("glue", "sst2")

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_a = datum['sentence']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, label=self.get_label_id(label))
            examples.append(example)
        return examples


class MNLIProcessor(DataProcessor):
    def __init__(self, subset):
        super().__init__()
        self.labels = [0, 1, 2]
        self.dataset = load_dataset("glue", subset)

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_b = datum['hypothesis']
            text_a = datum['premise']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_label_id(label))
            examples.append(example)
        return examples


class QNLIProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
        self.dataset = load_dataset("glue", "qnli")

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_b = datum['sentence']
            text_a = datum['question']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_label_id(label))
            examples.append(example)
        return examples


class COLAProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
        self.dataset = load_dataset("glue", "cola")

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_a = datum['sentence']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, label=self.get_label_id(label))
            examples.append(example)
        return examples


class MRPCProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
        self.dataset = load_dataset("glue", "mrpc")

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_a = datum['sentence1']
            text_b = datum['sentence2']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_label_id(label))
            examples.append(example)
        return examples


class QQPProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
        self.dataset = load_dataset("glue", "qqp")

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_a = datum['question1']
            text_b = datum['question2']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_label_id(label))
            examples.append(example)
        return examples


class RTEProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 1]
        self.dataset = load_dataset("glue", "rte")

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_a = datum['sentence1']
            text_b = datum['sentence2']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=self.get_label_id(label))
            examples.append(example)
        return examples


class STSBProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [0, 5]
        self.dataset = load_dataset("glue", "stsb")

    def get_examples(self, split):
        data = self.dataset[split]
        examples = []
        for datum in data:
            text_a = datum['sentence1']
            text_b = datum['sentence2']
            label = datum['label']
            guid = datum['idx']
            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
        return examples


PROCESSORS = {
    "sst2": SST2Processor,
    "mnli": MNLIProcessor,
    "qnli": QNLIProcessor,
    "cola": COLAProcessor,
    "mrpc": MRPCProcessor,
    "qqp": QQPProcessor,
    "rte": RTEProcessor,
    "stsb": STSBProcessor,
}
