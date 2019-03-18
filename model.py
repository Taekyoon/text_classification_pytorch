from typing import Tuple

from modules import DualWordEmbedding, ConvolutionLayer, MaxOverTimePoolLayer, FeedForwardLayer

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class TextClassifier(Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dims: int,
                 conv_in_channels: int,
                 conv_out_channels: int,
                 num_classes: int):
        super(TextClassifier, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_dims = embedding_dims
        self._in_channels = conv_in_channels
        self._out_channels = conv_out_channels
        self._num_classes = num_classes

        self.dual_embedding = DualWordEmbedding(self._vocab_size, self._embedding_dims)
        self.conv_layer = ConvolutionLayer(self._in_channels, self._out_channels)
        self.max_pool_layer = MaxOverTimePoolLayer()
        self.output_linear_layer = FeedForwardLayer(self._out_channels, self._num_classes)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        _inputs = inputs

        static_embeddings, non_static_embeddings = self.dual_embedding(inputs)

        static_tri_grams, static_tetra_grams, static_penta_grams = self.conv_layer(static_embeddings)
        non_static_tri_grams, non_static_tetra_grams, non_static_penta_grams = self.conv_layer(non_static_embeddings)

        combined_tri_grams = static_tri_grams + non_static_tri_grams
        combined_tetra_grams = static_tetra_grams + non_static_penta_grams
        combined_penta_grams = static_penta_grams + non_static_penta_grams

        max_pool_tri_grams = self.max_pool_layer(combined_tri_grams)
        max_pool_tetra_grams = self.max_pool_layer(combined_tetra_grams)
        max_pool_penta_grams = self.max_pool_layer(combined_penta_grams)

        max_pool_features = torch.cat((max_pool_tri_grams, max_pool_tetra_grams, max_pool_penta_grams), 1)

        output_features = self.output_linear_layer(max_pool_features)
        output_scores = F.log_softmax(output_features)

        return output_features, output_scores
