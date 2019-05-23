from typing import List, Any, Tuple, NewType, Dict
import itertools

from konlpy.tag import Okt

from tqdm import tqdm
import numpy as np
import pandas as pd

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import gluonnlp as nlp

from sklearn.model_selection import train_test_split

TextPolarityDataManager = NewType('TextPolarityDataManager', object)
TextPolarityDataset = NewType('TextPolarityDataset', Dataset)


def read_csv_dataset(path: str) -> List:
    data = pd.read_csv(path, sep="\t")
    data = data.dropna()

    return data


class TextPolarityDataManager(object):

    class TextPolarityDataset(Dataset):
        def __init__(self, dataset: Tuple[List, List, Any], has_len: bool = True) -> None:
            self._dataset = dataset
            self._has_len = has_len

            return

        def __len__(self) -> int:
            len_dataset = len(self._dataset)

            return len_dataset

        def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
            _input_data = self._dataset[0][idx]
            _label_data = self._dataset[1][idx]

            if self._has_len:
                _input_len = self._dataset[2][idx]

            input_tensor = Tensor(_input_data)
            label_tensor = Tensor([_label_data])
            input_len = Tensor(_input_len)

            return input_tensor, label_tensor, input_len

    def __init__(self, train_path: str, test_path: str, limit_len: int = 200, bucketing: bool = True) -> None:
        self._train_path = train_path
        self._test_path = test_path

        self._morph_analyzer = None
        self._tokenizer = None

        self._raw_dataset = None

        self._train_dataset = None
        self._test_dataset = None

        self.limit_len = limit_len
        self.bucketing = bucketing

        return

    def load_dataset(self) -> TextPolarityDataManager:
        _train_path, _test_path = self._train_path, self._test_path

        self._raw_dataset = self._load_dataset(_train_path, _test_path)

        return self

    def morphalize(self) -> TextPolarityDataManager:
        _train_text_dataset = self._raw_dataset['train']['text']
        _test_text_dataset = self._raw_dataset['test']['text']
        self._morph_analyzer = self._crate_morph_analyzer()

        _train_morphed_text_dataset = self._trans_morph_spaced_dataset(_train_text_dataset, self._morph_analyzer)
        _test_morphed_text_dataset = self._trans_morph_spaced_dataset(_test_text_dataset, self._morph_analyzer)

        self._raw_dataset['train']['morph_text'] = _train_morphed_text_dataset
        self._raw_dataset['test']['morph_text'] = _test_morphed_text_dataset

        return self

    def char_tokenize(self) -> TextPolarityDataManager:
        _train_text_dataset = self._raw_dataset['train']['text']
        _test_text_dataset = self._raw_dataset['test']['text']

        _train_morphed_text_dataset = self._trans_morph_spaced_dataset(_train_text_dataset)
        _test_morphed_text_dataset = self._trans_morph_spaced_dataset(_test_text_dataset)

        self._raw_dataset['train']['morph_text'] = _train_morphed_text_dataset
        self._raw_dataset['test']['morph_text'] = _test_morphed_text_dataset

        return self

    def build_tokenizer(self) -> TextPolarityDataManager:
        _train_morph_text_dataset = self._raw_dataset['train']['morph_text']

        tokenizer = self._build_tokenizer(_train_morph_text_dataset)

        self._tokenizer = tokenizer

        return self

    def preprocess(self) -> TextPolarityDataManager:
        _train_morph_text_dataset = self._raw_dataset['train']['morph_text']
        _train_label_dataset = self._raw_dataset['train']['label']
        _test_morph_text_dataset = self._raw_dataset['test']['morph_text']
        _test_label_dataset = self._raw_dataset['test']['label']
        _tokenizer = self._tokenizer

        numerized_train_text_dataset = self._preprocess_input_dataset(_train_morph_text_dataset, _tokenizer,
                                                                      limit_seq_len=self.limit_len)
        numerized_test_text_datset = self._preprocess_input_dataset(_test_morph_text_dataset, _tokenizer,
                                                                    limit_seq_len=self.limit_len)

        train_input_lens = self._get_input_lengths(_train_morph_text_dataset, limit_seq_len=self.limit_len)
        test_input_lens = self._get_input_lengths(_test_morph_text_dataset, limit_seq_len=self.limit_len)

        self._train_dataset = self._from_tensor_slices(numerized_train_text_dataset, _train_label_dataset,
                                                       lens=train_input_lens)
        self._test_dataset = self._from_tensor_slices(numerized_test_text_datset, _test_label_dataset,
                                                      lens=test_input_lens)

        return self

    def get_train_data_loader(self, batch_size: int, drop_last: bool = True) -> DataLoader:
        _batch_size = batch_size
        _drop_last = drop_last
        _train_torch_dataset = self._get_train_dataset()

        train_data_loader = DataLoader(_train_torch_dataset,
                                       batch_size=_batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=_drop_last)

        return train_data_loader

    def get_test_data_loader(self, batch_size: int=1) -> DataLoader:
        _batch_size = batch_size
        _test_torch_dataset = self._get_test_dataset()

        test_data_loader = DataLoader(_test_torch_dataset,
                                      batch_size=_batch_size)

        return test_data_loader

    def get_pretrained_word_embedding(self):
        _tokenizer = self._tokenizer

        pretrinaed_embedding = self._create_pretrained_embedding(_tokenizer)

        return pretrinaed_embedding

    def get_vocabulary_size(self):
        return len(self._tokenizer.token_to_idx)

    def _from_tensor_slices(self, inputs, labels, lens=None):
        _inputs = inputs
        _labels = labels
        _lens = lens

        if lens is not None:
            dataset = [(i, l, n) for i, l, n in zip(_inputs, _labels, _lens)]
        else:
            dataset = [(i, l) for i, l in zip(_inputs, _labels)]

        return dataset

    def _get_train_dataset(self) -> TextPolarityDataset:
        train_dataset = TextPolarityDataset(self._train_dataset)

        return train_dataset

    def _get_test_dataset(self) -> TextPolarityDataset:
        test_dataset = TextPolarityDataset(self._test_dataset)

        return test_dataset

    def _load_dataset(self, train_path, test_path, load_type: str='csv') -> Dict:
        _train_path = train_path
        _test_path = test_path

        if load_type == 'csv':
            _raw_train_data = read_csv_dataset(_train_path)
            _raw_test_data = read_csv_dataset(_test_path)

            train_text_data = self._extract_text_data(_raw_train_data)
            train_label_data = self._extract_label_data(_raw_train_data)

            test_text_data = self._extract_text_data(_raw_test_data)
            test_label_data = self._extract_label_data(_raw_test_data)
        else:
            raise NotImplementedError()

        return {'train': {'text': train_text_data,
                          'label': train_label_data},
                'test': {'text': test_text_data,
                         'label': test_label_data}}

    def _extract_text_data(self, dataset: pd.DataFrame) -> List:
        _dataset = dataset

        text_dataset = list(_dataset['document'])

        return text_dataset

    def _extract_label_data(self, dataset: pd.DataFrame) -> List:
        _dataset = dataset

        label_dataset = list(_dataset['label'])

        return label_dataset

    def _crate_morph_analyzer(self) -> Any:
        morph_analyzer = Okt()

        return morph_analyzer

    def _trans_morph_spaced_sentence(self, text: str, morph_analyzer: Any) -> str:
        _text = text
        morph_tokenized_text = ' '.join(morph_analyzer.morphs(_text.replace(' ', '')))

        return morph_tokenized_text

    def _trans_morph_spaced_dataset(self, data: List) -> List:
        _data = data
        morph_tokenized_data = list()

        for row in tqdm(_data):
            morph_tokenized_data.append(' '.join([s for s in row.replace(' ', '')]))

        return morph_tokenized_data

    def _preprocess_input_dataset(self, input_dataset: List, tokenizer: nlp.Vocab, limit_seq_len: int = 200) -> np.array:
        _input_dataset = input_dataset
        _tokenizer = tokenizer

        tokenized_dataset = [sent.split() for sent in _input_dataset]
        indicied_dataset = [_tokenizer.to_indices(tokens) for tokens in tokenized_dataset]
        processed_data = pad_sequences(indicied_dataset, maxlen=limit_seq_len, padding='post',
                                       truncating='post', value=1)

        return processed_data

    def _get_input_lengths(self, input_dataset: List, limit_seq_len: int = 200) -> np.array:
        _input_dataset = input_dataset

        input_lengths = [len(sent.split()) if len(sent.split()) <= limit_seq_len else limit_seq_len for sent in _input_dataset ]
        input_lengths = np.array(input_lengths, dtype=np.int64)

        return input_lengths

    def _build_tokenizer(self, data: List) -> nlp.Vocab:
        _text_dataset = data

        tokenized_dataset = [sent.split() for sent in _text_dataset]
        counter = nlp.data.count_tokens(itertools.chain.from_iterable([tokens for tokens in tokenized_dataset]))
        tokenizer = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

        return tokenizer

    def _create_pretrained_embedding(self, tokenizer):
        _tokenizer = tokenizer

        ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
        _tokenizer.set_embedding(ptr_embedding)
        pretrained_embedding = _tokenizer.embedding.idx_to_vec.asnumpy()

        return pretrained_embedding

    def _get_train_and_test_dataset(self, input_dataset: List, label_dataset: List) -> Tuple[Tuple, Tuple]:
        train_input, test_input, train_label, test_label = train_test_split(input_dataset, label_dataset,
                                                                            test_size=0.1, random_state=12223)

        return (train_input, train_label), (test_input, test_label)
