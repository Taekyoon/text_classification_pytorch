from typing import Tuple

import numpy as np

import torch
from torch import Tensor
from torch.nn import Embedding, Conv1d, Linear, BatchNorm1d, MaxPool1d, ReLU, AdaptiveAvgPool1d
from torch.nn import Module
from torch.nn import functional as F


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

        self.embedding = Embedding(self._vocab_size, self._embedding_dim)

        if embeddings is not None:
            print('has embeddings ==> ', embeddings.shape)
            self.embedding.weight.data.copy_(torch.Tensor(embeddings))

    def forward(self, *inputs) -> Tensor:
        _inputs = inputs[0]

        embeddings = self.embedding(_inputs)

        return embeddings


class TemporalConvNet(Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(TemporalConvNet, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self._conv_net = Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1)

    def forward(self, *inputs):
        _inputs = inputs[0]

        outputs = self._conv_net(_inputs)

        return outputs


class ConvolutionBlock(Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ConvolutionBlock, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._equality = self._in_channels == self._out_channels

        self._temp_conv_1 = Conv1d(in_channels=self._in_channels,
                                   out_channels=self._out_channels,
                                   kernel_size=3,
                                   padding=1)

        self._temp_conv_2 = Conv1d(in_channels=self._out_channels,
                                   out_channels=self._out_channels,
                                   kernel_size=3,
                                   padding=1)

        self._temp_batch_norm_1 = BatchNorm1d(self._out_channels)
        self._temp_batch_norm_2 = BatchNorm1d(self._out_channels)

        if not self._equality:
            self._residual_conv = Conv1d(in_channels=self._in_channels,
                                         out_channels=self._out_channels,
                                         kernel_size=1)

            self._residual_batch_norm = BatchNorm1d(self._out_channels)

    def forward(self, *inputs):
        _inputs = inputs[0]

        temp_conv_1 = self._temp_conv_1(_inputs)
        temp_conv_1 = self._temp_batch_norm_1(temp_conv_1)
        temp_conv_1 = F.relu(temp_conv_1)

        temp_conv_2 = self._temp_conv_2(temp_conv_1)
        temp_conv_2 = self._temp_batch_norm_2(temp_conv_2)
        temp_conv_2 = F.relu(temp_conv_2)

        if not self._equality:
            residual_conv = self._residual_conv(_inputs)
            residual_conv = self._residual_batch_norm(residual_conv)
        else:
            residual_conv = _inputs

        outputs = temp_conv_2 + residual_conv
        outputs = F.relu(outputs)

        return outputs


class HalfMaxPool(Module):
    def __init__(self):
        super(HalfMaxPool, self).__init__()

        self._max_pool = MaxPool1d(3, stride=2, padding=1)

    def forward(self, *inputs):
        _inputs = inputs[0]

        max_pooled = self._max_pool(_inputs)

        return max_pooled


class KMaxPooling(Module):
    def __init__(self,
                 k=8):
        super(KMaxPooling, self).__init__()

        self._k = k

        self._k_max_pool = AdaptiveAvgPool1d(self._k)

    def forward(self, *inputs):
        _inputs = inputs[0]

        max_pooled = self._k_max_pool(_inputs)

        return max_pooled


class FeedForwardLayer(Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int,
                 activation=ReLU()) -> None:
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
