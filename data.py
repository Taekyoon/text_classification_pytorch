from typing import List, Any, Tuple, NewType, Dict

from konlpy.tag import Okt

from tqdm import tqdm
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

TextPolarityPreprocessor = NewType('TextPolarityPreprocessor', object)
TextPolarityDataset = NewType('TextPolarityDataset', Dataset)


def read_csv_dataset(path: str) -> List:
    data = pd.read_csv(path)

    return data


class TextPolarityPreprocessor(object):

    class TextPolarityDataset(Dataset):
        def __init__(self, dataset: Tuple[List, List]) -> None:
            self._dataset = dataset

            return

        def __len__(self) -> int:
            len_dataset = len(self._dataset)

            return len_dataset

        def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
            _input_data = self._dataset[0][idx]
            _label_data = self._dataset[1][idx]

            input_tensor = Tensor(_input_data)
            label_tensor = Tensor([_label_data])

            return input_tensor, label_tensor

    def __init__(self, train_path: str, test_path: str) -> None:
        self._train_path = train_path
        self._test_path = test_path

        self._morph_analyzer = None
        self._toknizer = None

        self._raw_dataset = None

        self._train_dataset = None
        self._test_datset = None

        return

    def load_dataset(self) -> TextPolarityPreprocessor:
        _train_path, _test_path = self._train_path, self._test_path

        self._raw_dataset = self._load_dataset(_train_path, _test_path)

        return self

    def morphalize(self) -> TextPolarityPreprocessor:
        _train_text_dataset = self._raw_dataset['train']['text']
        _test_text_dataset = self._raw_dataset['test']['text']
        self._morph_analyzer = self._crate_morph_analyzer()

        _train_morphed_text_dataset = self._trans_morph_spaced_dataset(_train_text_dataset, self._morph_analyzer)
        _test_morphed_text_dataset = self._trans_morph_spaced_dataset(_test_text_dataset, self._morph_analyzer)

        self._raw_dataset['train']['morph_text'] = _train_morphed_text_dataset
        self._raw_dataset['test']['morph_text'] = _test_morphed_text_dataset

        return self

    def build_tokenizer(self) -> TextPolarityPreprocessor:
        _train_morph_text_dataset = self._raw_dataset['train']['morph_text']

        tokenizer = self._build_tokenizer(_train_morph_text_dataset)

        self._toknizer = tokenizer

        return self

    def preprocess(self) -> TextPolarityPreprocessor:
        _train_morph_text_dataset = self._raw_dataset['train']['morph_text']
        _train_label_dataset = self._raw_dataset['train']['label']
        _test_morph_text_dataset = self._raw_dataset['test']['morph_text']
        _test_label_dataset = self._raw_dataset['test']['label']
        _tokenizer = self._toknizer

        numerized_train_text_dataset = self._preprocess_input_dataset(_train_morph_text_dataset, _tokenizer)
        numerized_test_text_datset = self._preprocess_input_dataset(_test_morph_text_dataset, _tokenizer)

        self._train_dataset = (numerized_train_text_dataset, _train_label_dataset)
        self._test_datset = (numerized_test_text_datset, _test_label_dataset)

        return self

    def get_train_dataset(self) -> TextPolarityDataset:
        train_dataset = TextPolarityDataset(self._train_dataset)

        return train_dataset

    def get_test_dataset(self) -> TextPolarityDataset:
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

    def _trans_morph_spaced_dataset(self, data: List, morph_analyzer: Any) -> List:
        _data = data
        morph_tokenized_data = list()

        for row in tqdm(_data):
            morph_tokenized_data.append(self._trans_morph_spaced_sentence(row, morph_analyzer))

        return morph_tokenized_data

    def _preprocess_input_dataset(self, input_dataset: List, tokenizer: Tokenizer, limit_seq_len=40) -> np.array:
        _input_dataset = input_dataset
        _char_tokenizer = tokenizer

        _input_dataset = _char_tokenizer.texts_to_sequences(_input_dataset)

        processed_data = pad_sequences(_input_dataset, maxlen=limit_seq_len, padding='post', truncating='post')

        return processed_data

    def _build_tokenizer(self, data: List) -> Tokenizer:
        _text_dataset = data

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(_text_dataset)

        return tokenizer

    def _get_train_and_test_dataset(self, input_dataset: List, label_dataset: List) -> Tuple[Tuple, Tuple]:
        train_input, test_input, train_label, test_label = train_test_split(input_dataset, label_dataset,
                                                                            test_size=0.1, random_state=12223)

        return (train_input, train_label), (test_input, test_label)
