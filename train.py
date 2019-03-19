from torch.utils.data import DataLoader
from torch.nn.modules import Module

from configs import *
from data import TextPolarityDataManager
from model import TextClassifier


def prepare_dataset() -> DataLoader:
    _train_path = TRAIN_DATASET_PATH
    _test_path = TEST_DATASET_PATH
    _batch_size = BATCH_SIZE

    data_manager = TextPolarityDataManager(_train_path, _test_path)

    data_manager.load_dataset()  \
                .morphalize()  \
                .build_tokenizer() \
                .preprocess()

    train_dataloader = data_manager.get_train_dataloader(_batch_size)
    test_dataloader = data_manager.get_test_dataloader(_batch_size)

    return train_dataloader, test_dataloader


def build_model(vocab_size, embedding_dims, conv_channels) ->  Module:
    _vocab_size = vocab_size
    _embedding_dims = embedding_dims
    _conv_channels = conv_channels

    model = TextClassifier(vocab_size,
                           embedding_dims,
                           conv_channels,
                           2)

    return model


class TrainManager(object):
    def __init__(self):
        self._train_data = None
        self._test_data = None
        self._model = None
        self._optimizer = None

        self._loss_fn = None
        self._metric_fn = None

        self._epochs = None

    def train(self):
        pass

    def register_device_configs(self):
        pass

    def _eval(self):
        pass

    def _train_epoch(self):
        pass
