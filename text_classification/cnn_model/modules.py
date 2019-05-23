from typing import Tuple

import numpy as np

import torch
from torch import Tensor
from torch.nn import Embedding, Conv1d, Linear
from torch.nn import Module
from torch.nn import functional as F


class DualWordEmbedding(Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 embeddings: np.array = None) -> None:
        super(DualWordEmbedding, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim

        self.static_embedding = Embedding(self._vocab_size, self._embedding_dim)
        self.non_static_embedding = Embedding(self._vocab_size, self._embedding_dim)

        if embeddings is not None:
            print('has embeddings ==> ', embeddings.shape)
            self.static_embedding.weight.data.copy_(torch.Tensor(embeddings))
            self.non_static_embedding.weight.data.copy_(torch.Tensor(embeddings))

        self.static_embedding.weight.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        _inputs = inputs

        static_embeddings = self.static_embedding(_inputs)
        non_static_embeddings = self.non_static_embedding(_inputs)

        return static_embeddings, non_static_embeddings


class ConvolutionLayer(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int) -> None:
        super(ConvolutionLayer, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels

        self.tri_gram_conv_net = Conv1d(in_channels=self._in_channels,
                                        out_channels=self._out_channels,
                                        kernel_size=3)
        self.tetra_gram_conv_net = Conv1d(in_channels=self._in_channels,
                                          out_channels=self._out_channels,
                                          kernel_size=4)
        self.penta_gram_conv_net = Conv1d(in_channels=self._in_channels,
                                          out_channels=self._out_channels,
                                          kernel_size=5)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        _inputs = inputs
        _inputs = _inputs.transpose(2, 1)

        conv_trigram_outputs = F.relu(self.tri_gram_conv_net(_inputs)).transpose(2, 1)
        conv_tetragram_outputs = F.relu(self.tetra_gram_conv_net(_inputs)).transpose(2, 1)
        conv_pentaagram_outputs = F.relu(self.penta_gram_conv_net(_inputs)).transpose(2, 1)

        return conv_trigram_outputs, conv_tetragram_outputs, conv_pentaagram_outputs


class MaxOverTimePoolLayer(Module):
    def __init__(self) -> None:
        super(MaxOverTimePoolLayer, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        _inputs = inputs

        max_pool_outputs = torch.max(_inputs, 1)[0]

        return max_pool_outputs


class FeedForwardLayer(Module):
    def __init__(self,
                 input_dims: int,
                 output_dims: int) -> None:
        super(FeedForwardLayer, self).__init__()

        self._input_dims = input_dims
        self._output_dims = output_dims

        self.linear = Linear(self._input_dims, self._output_dims)

    def forward(self, inputs: Tensor) -> Tensor:
        _inputs = inputs
        linear_outputs = self.linear(inputs)

        return linear_outputs
