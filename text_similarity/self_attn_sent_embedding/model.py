import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F

from modules import BiRNN, Attention, FeedForwardLayer


class SelfAttentionNetwork(Module):
    def __init__(self,
                 rnn_hidden_dims,
                 attn_hidden_dims,
                 hops,
                 word_embedding):
        super(SelfAttentionNetwork, self).__init__()

        self._rnn_hidden_dims = rnn_hidden_dims
        self._attn_hidden_dims = attn_hidden_dims
        self._device = None

        self.embedding_dims = self._rnn_hidden_dims * 2
        self.hops = hops

        self._embedding = word_embedding
        self._embedding_dims = self._embedding._embedding_dim

        self._birnn = BiRNN(self._embedding_dims, self._rnn_hidden_dims)
        self._attention = Attention(self._rnn_hidden_dims * 2, self._attn_hidden_dims, self.hops)

    def to(self, device):
        super(SelfAttentionNetwork, self).to(device)
        self._device = device

    def forward(self, *inputs):
        _input_layer = inputs[0]
        _batch_size = _input_layer.size(0)

        _lens = None

        if len(inputs) > 1:
            _lens = inputs[1]

        word_embedding_layer = self._embedding(_input_layer)

        hidden_state = self._birnn.get_init_state(_batch_size, device=1)
        rnn_output_layer, _ = self._birnn(word_embedding_layer, hidden_state, _lens)

        attention_layer, attention_scores = self._attention(rnn_output_layer, _lens)

        return attention_layer, attention_scores


class SentenceSimilarityModel(Module):
    def __init__(self,
                 base_encoder,
                 target_encoder,
                 hidden_dims,
                 num_classes):
        super(SentenceSimilarityModel, self).__init__()

        self._base_encoder = base_encoder
        self._target_encoder = target_encoder

        self._sent_embedding_dims = base_encoder.embedding_dims
        self._attention_hops = base_encoder.hops
        self._hidden_dims = hidden_dims
        self._num_classes = num_classes

        self._weight_base = FeedForwardLayer(self._sent_embedding_dims, self._sent_embedding_dims, bias=False,
                                             activation=None)
        self._weight_target = FeedForwardLayer(self._sent_embedding_dims, self._sent_embedding_dims, bias=False,
                                               activation=None)

        self._fully_connected_1 = FeedForwardLayer(self._sent_embedding_dims * self._attention_hops, self._hidden_dims)
        self._fully_connected_2 = FeedForwardLayer(self._hidden_dims, self._num_classes, activation=None)

    def forward(self, *inputs):
        _base_sentence, _target_sentence = inputs[0], inputs[1]
        _batch_size = _base_sentence.size()[0]

        if len(inputs) > 2:
            _lens_pair = inputs[2]

        base_sent_embedding, base_attn = self._base_encoder(_base_sentence, _lens_pair[0])
        target_sent_embedding, target_attn = self._target_encoder(_target_sentence, _lens_pair[1])

        weighted_base_sent_embedding = self._weight_base(base_sent_embedding)
        weighted_target_sent_embedding = self._weight_target(target_sent_embedding)

        element_producted_layer = weighted_base_sent_embedding * weighted_target_sent_embedding
        element_producted_layer = element_producted_layer.view(_batch_size, -1)
        hidden_layer = self._fully_connected_1(element_producted_layer)
        output_layer = self._fully_connected_2(hidden_layer)

        return output_layer, base_attn, target_attn
