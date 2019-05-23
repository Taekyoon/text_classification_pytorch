from typing import Tuple, List

from modules import CharacterEmbedding, TemporalConvNet, ConvolutionBlock, \
                    HalfMaxPool, FeedForwardLayer, KMaxPooling,\
                    transpose_sequence_and_feature_dims, log_softmax, flatten

from torch import Tensor
from torch.nn import Module, ModuleList


class TextClassifier(Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dims: int,
                 conv_block_in_channels: List,
                 conv_block_out_channels: List,
                 linear_in_channels: List,
                 linear_out_channels: List,
                 k_max_pool:int,
                 num_classes: int,
                 embeddings=None):
        super(TextClassifier, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_dims = embedding_dims
        self._conv_block_in_channels = conv_block_in_channels
        self._conv_block_out_channels = conv_block_out_channels
        self._linear_in_channels = linear_in_channels
        self._linear_out_channels = linear_out_channels
        self._k_maxpool = k_max_pool
        self._num_classes = num_classes
        self._embeddings = embeddings

        self._char_embedding = CharacterEmbedding(self._vocab_size,
                                                  self._embedding_dims,
                                                  embeddings=embeddings)
        self._temp_conv = TemporalConvNet(self._embedding_dims,
                                          self._conv_block_in_channels[0])

        self._conv_blocks = ModuleList([ConvolutionBlock(i, o) for i, o in zip(self._conv_block_in_channels,
                                                                               self._conv_block_out_channels)])

        self._half_maxpool = HalfMaxPool()

        self._k_maxpool = KMaxPooling(self._k_maxpool)

        self._fully_connected = ModuleList([FeedForwardLayer(i, o) for i, o in zip(self._linear_in_channels,
                                                                                   self._linear_out_channels)])

        self._output_linear = FeedForwardLayer(self._linear_out_channels[-1], self._num_classes,
                                               activation=None)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        _input_layer = inputs

        embedding_layer = self._char_embedding(_input_layer)

        embedding_layer = transpose_sequence_and_feature_dims(embedding_layer)
        temp_conv_layer = self._temp_conv(embedding_layer)

        conv_input = temp_conv_layer
        conv_final_output = None

        for i, conv_block in enumerate(self._conv_blocks):
            conv_output = conv_block(conv_input)

            if i == len(self._conv_blocks) - 1:
                conv_final_output = self._k_maxpool(conv_output)
                break
            elif i % 2 == 1:
                conv_output = self._half_maxpool(conv_output)

            conv_input = conv_output

        linear_input = flatten(conv_final_output)
        linear_final_output = None

        for i, feedforward in enumerate(self._fully_connected):
            linear_output = feedforward(linear_input)

            if i == len(self._fully_connected) - 1:
                linear_final_output = linear_output
                break

            linear_input = linear_output

        output_features = self._output_linear(linear_final_output)
        output_scores = log_softmax(output_features)

        return output_features, output_scores
