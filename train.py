import torch
from torch.utils.data import DataLoader
from torch.nn.modules import Module

from tqdm import tqdm

from typing import Dict

from data import TextPolarityDataManager
from model import TextClassifier


def prepare_dataset(configs: Dict) -> DataLoader:
    _train_path = configs['dataset']['train_path']
    _test_path = configs['dataset']['test_path']

    data_manager = TextPolarityDataManager(_train_path, _test_path)

    data_manager.load_dataset()  \
                .morphalize()  \
                .build_tokenizer() \
                .preprocess()

    return data_manager


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
    def __init__(self,
                 train_data_loader,
                 test_data_loader,
                 model,
                 epochs,
                 eval_steps,
                 learning_rate = 0.001,
                 loss_fn=torch.nn.CrossEntropyLoss,
                 optimizer=torch.optim.Adam,
                 metric_fn=None):

        self._epochs = epochs
        self._eval_steps = eval_steps
        self._learning_rate = learning_rate

        self._train_data_loader = train_data_loader
        self._test_data_loader = test_data_loader
        self._model = model
        self._optimizer = optimizer(params=self._model.parameters(), lr=self._learning_rate)

        self._loss_fn = loss_fn()
        self._metric_fn = metric_fn

    def train(self):
        for i in range(self._epochs):
            self._train_epoch()

    def register_device_configs(self):
        pass

    def _eval(self):
        model = self._model
        loss_fn = self._loss_fn
        data_loader = self._test_data_loader

        total, match = 0, 0
        loss = 0.

        model.eval()

        for step, sampled_batch in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
            sample_input = sampled_batch[0].long()
            sample_label = sampled_batch[1].long()

            output_features, output_scores = model(sample_input)

            loss += loss_fn(output_scores, sample_label).item()

            predictions = torch.max(output_features, -1)[1]

            for p, t in zip(predictions, sample_label):
                total += 1
                if p == t:
                    match += 1
        else:
            loss /= (step + 1)
            accuracy = match / total

        return accuracy, loss

    def _train_epoch(self):
        model = self._model
        loss_fn = self._loss_fn
        data_loader = self._train_data_loader

        steps_in_epoch = len(data_loader)
        tr_loss = 0

        model.train()

        for step, sample_batched in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
            sample_input = sample_batched[0].long()
            sample_label = sample_batched[1].long()

            _, output_scores = model(sample_input)
            loss = loss_fn(output_scores, sample_label)
            tr_loss = loss.item()
            self._backprop(loss)

            if (self._epochs * steps_in_epoch + step) % self._eval_steps:
                val_acc, val_loss = self._eval()
                model.train()
        else:
            tr_loss /= (step + 1)

        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_acc : {:.3f}, val_loss : {:.3f}'.format(self._epochs + 1,
                                                                                              tr_loss, val_acc,
                                                                                              val_loss))

    def _backprop(self, loss) -> None:
        optimizer = self._optimizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
