import numpy as np

import torch
from torch import Tensor
from torch.nn import Embedding, Conv1d, Linear, MaxPool1d, GRU
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def log_softmax(tensors):
    return F.log_softmax(tensors, dim=-1)


def flatten(tensors):
    batch_size = tensors.size()[0]
    return tensors.view(batch_size, -1)


def transpose_sequence_and_feature_dims(tensors):
    _tensors = tensors

    _tensors = _tensors.transpose(2, 1)

    return _tensors


class CharacterEmbedding(Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 embeddings: np.array = None) -> None:
        super(CharacterEmbedding, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim

        self.embedding = Embedding(self._vocab_size, self._embedding_dim, padding_idx=0)

        if embeddings is not None:
            print('has embeddings ==> ', embeddings.shape)
            self.embedding.weight.data.copy_(torch.Tensor(embeddings))

    def forward(self, *inputs) -> Tensor:
        _inputs = inputs[0]

        embeddings = self.embedding(_inputs)

        return embeddings


class ConvNet(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super(ConvNet, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size

        self._conv_net = Conv1d(in_channels=self._in_channels,
                                out_channels=self._out_channels,
                                kernel_size=self._kernel_size,
                                padding=self._kernel_size // 2)

    def forward(self, *inputs):
        _inputs = inputs[0]

        outputs = self._conv_net(_inputs)
        outputs = F.relu(outputs)

        return outputs


class MaxPool(Module):
    def __init__(self):
        super(MaxPool, self).__init__()

        self._max_pool = MaxPool1d(2, stride=2)

    def forward(self, *inputs):
        _inputs = inputs[0]

        max_pooled = self._max_pool(_inputs)

        return max_pooled


class BiRNN(Module):
    def __init__(self,
                 input_dims,
                 output_dims):
        super(BiRNN, self).__init__()

        self._input_dims = input_dims
        self._output_dims = output_dims

        self._bi_gru = GRU(self._input_dims, self._output_dims,
                           batch_first=True, bidirectional=True, dropout=0.5)

    def forward(self, *inputs, device=None):
        _inputs = inputs[0]
        _device = device

        if len(inputs) > 2:
            _lens = inputs[2]

            _lens, sorted_indices = torch.sort(_lens, descending=True)
            sorted_indices = sorted_indices.to(_device)
            _, restoration_indices = torch.sort(sorted_indices, descending=False)
            _inputs = _inputs.index_select(0, sorted_indices)
            _inputs = pack_padded_sequence(_inputs, _lens, batch_first=True)

            outputs, final_state = self._bi_gru(_inputs)

            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs.index_select(0, restoration_indices.to(_device))
            final_state = final_state.index_select(1, restoration_indices.to(_device))
        else:
            _hidden_state = inputs[1]
            outputs, final_state = self._bi_gru(_inputs, _hidden_state)

        final_state = final_state.transpose(0, 1).contiguous().view(-1, self._output_dims * 2)

        return outputs, final_state

    def get_init_state(self, batch_size, device=None):
        _device = device

        hidden_state = torch.zeros(2, batch_size, self._output_dims)

        return hidden_state.to(_device) if _device else hidden_state


class FeedForwardLayer(Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 activation=F.relu) -> None:
        super(FeedForwardLayer, self).__init__()

        self._input_dims = input_dims
        self._output_dims = output_dims

        self._linear = Linear(self._input_dims, self._output_dims)
        self._activation = activation

    def forward(self, *inputs) -> Tensor:
        _inputs = inputs[0]

        outputs = self._linear(_inputs)

        if self._activation:
            outputs = self._activation(outputs)

        return outputs
