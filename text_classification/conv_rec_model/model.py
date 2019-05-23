from typing import Tuple, List

from modules import CharacterEmbedding, ConvNet, MaxPool, \
                    FeedForwardLayer, BiRNN,\
                    transpose_sequence_and_feature_dims, log_softmax

from torch import Tensor
from torch.nn import Module, ModuleList, Dropout


class TextClassifier(Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dims: int,
                 conv_block_in_channels: List,
                 conv_block_out_channels: List,
                 conv_block_kernel_size: List,
                 bi_rnn_in_channels: List,
                 bi_rnn_out_channels: List,
                 num_classes: int,
                 embeddings=None):
        super(TextClassifier, self).__init__()

        self._vocab_size = vocab_size
        self._embedding_dims = embedding_dims
        self._conv_block_in_channels = conv_block_in_channels
        self._conv_block_out_channels = conv_block_out_channels
        self._conv_block_kernel_size = conv_block_kernel_size
        self._bi_rnn_in_channels = bi_rnn_in_channels
        self._bi_rnn_out_channels = bi_rnn_out_channels
        self._num_classes = num_classes
        self._embeddings = embeddings
        self._device = None

        self._char_embedding = CharacterEmbedding(self._vocab_size,
                                                  self._embedding_dims,
                                                  embeddings=embeddings)

        self._conv_blocks = ModuleList([ConvNet(i, o, k) for i, o, k in zip(self._conv_block_in_channels,
                                                                            self._conv_block_out_channels,
                                                                            self._conv_block_kernel_size)])

        self._max_pools = ModuleList([MaxPool() for _ in range(len(self._conv_blocks))])

        self._rnn_layers = ModuleList([BiRNN(i, o) for i, o in zip(self._bi_rnn_in_channels,
                                                                   self._bi_rnn_out_channels)])

        self._output_linear = FeedForwardLayer(self._bi_rnn_out_channels[-1] * 2, self._num_classes,
                                               activation=None)
        self._dropout = Dropout(0.5)

    def to(self, device):
        super(TextClassifier, self).to(device)
        self._device = device

    def forward(self, inputs: Tensor, lens: Tensor = None) -> Tuple[Tensor, Tensor]:
        _input_layer = inputs
        _batch_size = inputs.size()[0]

        if lens is not None:
            _lens = lens

        embedding_layer = self._char_embedding(_input_layer)

        embedding_layer = transpose_sequence_and_feature_dims(embedding_layer)

        conv_input_layer = embedding_layer

        for conv, max_pool in zip(self._conv_blocks, self._max_pools):
            conv_output_layer = conv(conv_input_layer)
            max_pool_layer = max_pool(conv_output_layer)

            conv_input_layer = max_pool_layer

        rnn_input_layer = transpose_sequence_and_feature_dims(max_pool_layer)
        rnn_input_layer = self._dropout(rnn_input_layer)

        for rnn in self._rnn_layers:
            if _lens is not None:
                _lens = Tensor([l // 4 if l // 4 > 1 else 1 for l in _lens]).to(self._device)
                rnn_output_layer, final_hidden_state = rnn(rnn_input_layer, None, _lens)
            else:
                hidden_state = rnn.get_init_state(_batch_size, device=self._device)
                rnn_output_layer, final_hidden_state = rnn(rnn_input_layer, hidden_state)

            rnn_input_layer = rnn_output_layer

        linear_input_layer = final_hidden_state

        output_features = self._output_linear(linear_input_layer)
        output_scores = log_softmax(output_features)

        return output_features, output_scores
