import torch
from torch.utils.data import DataLoader
from torch.nn.modules import Module

from tqdm import tqdm

from typing import Dict

from data import TextPolarityDataManager

from modules import WordEmbedding
from model import SelfAttentionNetwork, SentenceSimilarityModel

SEED_NUM = 7


def prepare_dataset(configs: Dict) -> DataLoader:
    _train_path = configs['dataset']['train_path']
    _test_path = configs['dataset']['test_path']

    data_manager = TextPolarityDataManager(_train_path, _test_path)

    data_manager.load_dataset() \
        .morphalize() \
        .build_tokenizer() \
        .preprocess()

    return data_manager


def build_model(configs, embeddings=None) -> Module:
    _vocab_size = configs['vocab_size']
    _embedding_dims = configs['embedding_size']
    _rnn_hidden_dims = configs['rnn_hidden_dims']
    _attn_hidden_dims = configs['attn_hidden_dims']
    _hops = configs['hops']
    _similarity_hidden_dims = configs['similarity_hidden_dims']
    _num_classes = configs['num_classes']

    _embeddings = embeddings

    word_embedding_model = WordEmbedding(_vocab_size, _embedding_dims, _embeddings)
    sent_embedding_model = SelfAttentionNetwork(_rnn_hidden_dims, _attn_hidden_dims, _hops, word_embedding_model)

    model = SentenceSimilarityModel(sent_embedding_model, sent_embedding_model, _similarity_hidden_dims, _num_classes)

    return model


def regularize(attn_mat, r, device):
    sim_mat = torch.bmm(attn_mat.permute(0, 2, 1), attn_mat)
    identity = torch.eye(r).to(device)

    p = torch.norm(sim_mat - identity, dim=(1, 2)).mean()
    return p


class TrainManager(object):
    def __init__(self,
                 train_data_loader,
                 test_data_loader,
                 model,
                 epochs,
                 eval_steps,
                 learning_rate=1e-4,
                 loss_fn=torch.nn.CrossEntropyLoss,
                 optimizer=torch.optim.Adam,
                 metric_fn=None,
                 gpu_device=1):

        self._epochs = epochs
        self._eval_steps = eval_steps
        self._learning_rate = learning_rate

        self._train_data_loader = train_data_loader
        self._test_data_loader = test_data_loader
        self._model = model
        self._optimizer = optimizer(params=self._model.parameters(), lr=self._learning_rate)

        self._device = torch.device('cuda:' + str(gpu_device)) if torch.cuda.is_available() \
                                                                  and gpu_device > 0 else torch.device('cpu')

        self._loss_fn = loss_fn()
        self._metric_fn = metric_fn

    def train(self):
        torch.manual_seed(SEED_NUM)
        for i in range(self._epochs):
            self._train_epoch(i)

    def register_device_configs(self):
        pass

    def _eval(self):
        model = self._model
        loss_fn = self._loss_fn
        data_loader = self._test_data_loader
        device = self._device

        total, match, loss = 0, 0, 0.

        model.eval()

        for step, sampled_batch in tqdm(enumerate(data_loader), desc='valid steps', total=len(data_loader)):
            sample_base_input = sampled_batch[0].long().to(device)
            sample_target_input = sampled_batch[1].long().to(device)
            sample_label = sampled_batch[2].long().to(device)
            sample_len_1 = sampled_batch[3].long().to(device)
            sample_len_2 = sampled_batch[4].long().to(device)

            outputs, _, _ = model(sample_base_input, sample_target_input, (sample_len_1, sample_len_2))

            loss += loss_fn(outputs, sample_label).item()
            predictions = torch.max(outputs, -1)[1]

            for p, t in zip(predictions, sample_label):
                total += 1
                if p == t:
                    match += 1
        else:
            loss /= (step + 1)
            accuracy = match / total

        return accuracy, loss

    def _train_epoch(self, epoch):
        model = self._model
        loss_fn = self._loss_fn
        data_loader = self._train_data_loader
        device = self._device

        steps_in_epoch = len(data_loader)
        tr_loss = 0
        val_loss = 0

        total, match = 0, 0

        model.to(device)
        model.train()

        for step, sampled_batch in tqdm(enumerate(data_loader), desc='train steps', total=len(data_loader)):
            sample_input_1 = sampled_batch[0].long().to(device)
            sample_input_2 = sampled_batch[1].long().to(device)
            sample_label = sampled_batch[2].long().to(device)
            sample_len_1 = sampled_batch[3].long().to(device)
            sample_len_2 = sampled_batch[4].long().to(device)

            outputs, base_attn_mat, target_attn_mat = model(sample_input_1, sample_input_2,
                                                            (sample_len_1, sample_len_2))

            predictions = torch.max(outputs, -1)[1]

            for p, t in zip(predictions, sample_label):
                total += 1
                if p == t:
                    match += 1

            loss = loss_fn(outputs, sample_label)
            tr_loss += loss.item()

            a_reg = regularize(base_attn_mat, 32, device)
            b_reg = regularize(target_attn_mat, 32, device)

            loss.add_(a_reg)
            loss.add_(b_reg)

            self._backprop(loss)

            if (epoch * steps_in_epoch + (step + 1)) % self._eval_steps == 0:
                val_acc, val_loss = self._eval()
                model.train()
        else:
            tr_acc = match / total
            tr_loss /= (step + 1)

        tqdm.write(
            'epoch : {}, tr_acc : {:.3f}, tr_loss : {:.3f}, val_acc : {:.3f}, val_loss : {:.3f}'.format(epoch + 1,
                                                                                                        tr_acc, tr_loss,
                                                                                                        val_acc, val_loss))

    def _backprop(self, loss) -> None:
        optimizer = self._optimizer

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _save_model(self):
        pass
