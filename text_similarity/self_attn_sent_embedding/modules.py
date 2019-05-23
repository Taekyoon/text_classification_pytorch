from typing import Tuple
import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.nn import GRU, Embedding, Linear


class WordEmbedding(Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 embeddings: np.array = None,
                 enable_grad: bool = True) -> None:
        super(WordEmbedding, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._enable_grad = enable_grad

        self._embedding = Embedding(self._vocab_size, self._embedding_dim, padding_idx=1)

        if embeddings is not None:
            print('has embeddings ==> ', embeddings.shape)
            self._embedding.weight.data.copy_(torch.Tensor(embeddings))

        self._embedding.weight.requires_grad_(self._enable_grad)

    def forward(self, *inputs: Tensor) -> Tuple[Tensor, Tensor]:
        _inputs = inputs[0]

        static_embeddings = self._embedding(_inputs)

        return static_embeddings


class BiRNN(Module):
    def __init__(self,
                 input_dims,
                 output_dims):
        super(BiRNN, self).__init__()

        self._input_dims = input_dims
        self._output_dims = output_dims

        self._bi_gru = GRU(self._input_dims, self._output_dims, dropout=0.5,
                           batch_first=True, bidirectional=True)

    def forward(self, *inputs, device=None):
        '''

        :param inputs:
        :param device:
        :return:
        '''
        _inputs = inputs[0]
        _device = device

        if False:
        # if len(inputs) > 2 and inputs[2] is not None:
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

        # final_state = final_state.transpose(0, 1).contiguous().view(-1, self._output_dims * 2)

        return outputs, final_state

    def get_init_state(self, batch_size, device=None):
        _device = device

        hidden_state = torch.zeros(2, batch_size, self._output_dims)

        return hidden_state.to(_device) if _device else hidden_state


class Attention(Module):
    def __init__(self,
                 input_dims,
                 hidden_dims,
                 hops):
        super(Attention, self).__init__()

        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._hops = hops

        self._linear_w1 = Linear(self._input_dims, self._hidden_dims)
        self._linear_w2 = Linear(self._hidden_dims, self._hops)

    def forward(self, *inputs):
        _inputs = inputs[0]
        _lens = None

        if len(inputs) > 1:
            _lens = inputs[1]
        hiddens = torch.tanh(self._linear_w1(_inputs))
        attention_weights = self._linear_w2(hiddens)

        if _lens is not None:
            seq_lens = torch.LongTensor(_lens.cpu())
            input_max_len = attention_weights.size(1)
            mask = torch.arange(input_max_len)[None, :] < seq_lens[:, None]
            attention_weights[~mask] = float('-inf')

        attention_scores = F.softmax(attention_weights, dim=1)
        transpose_attention_scores = torch.transpose(attention_scores, 2, 1)

        weighted_sum_vectors = torch.bmm(transpose_attention_scores, _inputs)

        return weighted_sum_vectors, attention_scores


class FeedForwardLayer(Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 activation=F.relu,
                 bias=True,
                 dropout=None) -> None:
        super(FeedForwardLayer, self).__init__()

        self._input_dims = input_dims
        self._output_dims = output_dims
        self._bias = bias
        self._dropout = dropout

        self._linear = Linear(self._input_dims, self._output_dims, bias=self._bias)
        self._activation = activation

    def forward(self, *inputs) -> Tensor:
        _inputs = inputs[0]

        outputs = self._linear(_inputs)

        if self._activation:
            outputs = self._activation(outputs)

        if self._dropout is not None:
            outputs = F.dropout(outputs, self._dropout)

        return outputs
