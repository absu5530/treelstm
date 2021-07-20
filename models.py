
from sklearn.metrics import accuracy_score, confusion_matrix

import mxnet as mx
from mxnet.gluon import Block, nn
from mxnet.gluon.parameter import Parameter
from transformers import BertModel
import gluonnlp
import numpy as np

class ChildSumLSTMCell(Block):
    def __init__(self, hidden_size,
                 i2h_weight_initializer=None,
                 hs2h_weight_initializer=None,
                 hc2h_weight_initializer=None,
                 i2h_bias_initializer='zeros',
                 hs2h_bias_initializer='zeros',
                 hc2h_bias_initializer='zeros',
                 input_size=0, prefix=None, params=None):
        super(ChildSumLSTMCell, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._hidden_size = hidden_size
            self._input_size = input_size
            self.i2h_weight = self.params.get('i2h_weight', shape=(4 * hidden_size, input_size),
                                              init=i2h_weight_initializer)
            self.hs2h_weight = self.params.get('hs2h_weight', shape=(3 * hidden_size, hidden_size),
                                               init=hs2h_weight_initializer)
            self.hc2h_weight = self.params.get('hc2h_weight', shape=(hidden_size, hidden_size),
                                               init=hc2h_weight_initializer)
            self.i2h_bias = self.params.get('i2h_bias', shape=(4 * hidden_size,),
                                            init=i2h_bias_initializer)
            self.hs2h_bias = self.params.get('hs2h_bias', shape=(3 * hidden_size,),
                                             init=hs2h_bias_initializer)
            self.hc2h_bias = self.params.get('hc2h_bias', shape=(hidden_size,),
                                             init=hc2h_bias_initializer)

    def forward(self, F, inputs, trees):

        forward_outputs = []
        for input_tree in zip(inputs, trees):
            forward_outputs.append(self.sample_forward(F, input_tree[0], input_tree[1])[1][1])

        return mx.nd.concat(*forward_outputs, dim=0)

    def sample_forward(self, F, input_doc, tree):
        children_outputs = [self.sample_forward(F, input_doc, child)
                            for child in tree.children]
        tree_vector = tree.vector

        if np.isnan(np.sum(tree_vector)):
            tree_vector = F.zeros_like(F.array(tree_vector))

        if children_outputs:
            _, children_states = zip(*children_outputs)  # unzip
        else:
            children_states = None

        with mx.cpu() as ctx:
            return self.node_forward(F, F.expand_dims(F.array(tree_vector), axis=0), children_states,
                                     self.i2h_weight.data(ctx),
                                     self.hs2h_weight.data(ctx),
                                     self.hc2h_weight.data(ctx),
                                     self.i2h_bias.data(ctx),
                                     self.hs2h_bias.data(ctx),
                                     self.hc2h_bias.data(ctx))

    def node_forward(self, F, input_node, children_states,
                     i2h_weight, hs2h_weight, hc2h_weight,
                     i2h_bias, hs2h_bias, hc2h_bias):
        # comment notation:
        # N for batch size
        # C for hidden state dimensions
        # K for number of children.

        # FC for i, f, u, o gates (N, 4*C), from input to hidden
        i2h = F.FullyConnected(data=input_node, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._hidden_size * 4)
        i2h_slices = F.split(i2h, num_outputs=4)  # (N, C)*4
        i2h_iuo = F.concat(*[i2h_slices[i] for i in [0, 2, 3]], dim=1)  # (N, C*3)

        if children_states:
            # sum of children states, (N, C)
            hs = F.add_n(*[state[0] for state in children_states])
            # concatenation of children hidden states, (N, K, C)
            hc = F.concat(*[F.expand_dims(state[0], axis=1) for state in children_states], dim=1)
            # concatenation of children cell states, (N, K, C)
            cs = F.concat(*[F.expand_dims(state[1], axis=1) for state in children_states], dim=1)

            # calculate activation for forget gate. addition in f_act is done with broadcast
            i2h_f_slice = i2h_slices[1]
            f_act = i2h_f_slice + hc2h_bias + F.dot(hc, hc2h_weight)  # (N, K, C)
            forget_gates = F.Activation(f_act, act_type='sigmoid')  # (N, K, C)
        else:
            # for leaf nodes, summation of children hidden states are zeros.
            hs = F.zeros_like(i2h_slices[0])

        # FC for i, u, o gates, from summation of children states to hidden state
        hs2h_iuo = F.FullyConnected(data=hs, weight=hs2h_weight, bias=hs2h_bias,
                                    num_hidden=self._hidden_size * 3)
        i2h_iuo = i2h_iuo + hs2h_iuo

        iuo_act_slices = F.SliceChannel(i2h_iuo, num_outputs=3)  # (N, C)*3
        i_act, u_act, o_act = iuo_act_slices[0], iuo_act_slices[1], iuo_act_slices[2]  # (N, C) each

        # calculate gate outputs
        in_gate = F.Activation(i_act, act_type='sigmoid')
        in_transform = F.Activation(u_act, act_type='tanh')
        out_gate = F.Activation(o_act, act_type='sigmoid')

        # calculate cell state and hidden state
        next_c = in_gate * in_transform
        if children_states:
            next_c = F.sum(forget_gates * cs, axis=1) + next_c
        next_h = out_gate * F.Activation(next_c, act_type='tanh')

        return next_h, [next_h, next_c]


class BertAndSimilarityTreeLSTM(nn.Block):
    def __init__(self, dropout_rate=0.5):
        super(BertAndSimilarityTreeLSTM, self).__init__()
        with self.name_scope():
            self.bert, _ = gluonnlp.model.get_model('bert_12_768_12',
                                                    dataset_name='book_corpus_wiki_en_uncased',
                                                    pretrained=True, use_pooler=True,
                                                    use_decoder=False, use_classifier=False)
            self.childsumtreelstm = ChildSumLSTMCell(256, input_size=768)
            #             self.similarity = Similarity(sim_hidden_size, rnn_hidden_size, num_classes)
            self.dropout_layer = nn.Dropout(rate=dropout_rate)
            self.dense_layer_1 = nn.Dense(128)
            self.dense_layer_2 = nn.Dense(64)
            self.dense_layer_3 = nn.Dense(16)
            self.dense_layer_4 = nn.Dense(2)

    def forward(self, F, token_ids, segment_ids, valid_length, l_sentences, l_trees, r_sentences, r_trees):
        _, bert_output = self.bert(token_ids, segment_ids, valid_length)

        bert_dropout_output = self.dropout_layer(bert_output)

        bert_dense_output = self.dense_layer_1(bert_dropout_output)

        l_output = self.childsumtreelstm(F, l_sentences, l_trees)

        l_dropout_output = self.dropout_layer(l_output)

        l_dense_output = self.dense_layer_2(l_dropout_output)

        r_output = self.childsumtreelstm(F, r_sentences, r_trees)

        r_dropout_output = self.dropout_layer(r_output)

        r_dense_output = self.dense_layer_2(r_dropout_output)

        lstm_concat = F.concat(l_dense_output, r_dense_output, dim=1)

        lstm_concat_dense_output = self.dense_layer_3(lstm_concat)

        bert_lstm_concat_output = F.concat(bert_dense_output, lstm_concat_dense_output, dim=1)

        bert_lstm_dropout_output = self.dropout_layer(bert_lstm_concat_output)

        final_output = self.dense_layer_4(bert_lstm_dropout_output)

        return final_output


class BertClassifier(nn.Block):
    def __init__(self, dropout_rate=0.5):
        super(BertClassifier, self).__init__()
        with self.name_scope():
            self.bert, _ = gluonnlp.model.get_model('bert_12_768_12',
                                                    dataset_name='book_corpus_wiki_en_uncased',
                                                    pretrained=True, use_pooler=True,
                                                    use_decoder=False, use_classifier=False)
            self.dropout_layer = nn.Dropout(rate=dropout_rate)
            # self.dense_layer_1 = nn.Dense(128)
            # self.dense_layer_2 = nn.Dense(64)
            # self.dense_layer_3 = nn.Dense(16)
            self.dense_layer_4 = nn.Dense(2)

    def forward(self, F, token_ids, segment_ids, valid_length, l_sentences, l_trees, r_sentences, r_trees):
        _, bert_output = self.bert(token_ids, segment_ids, valid_length)

        bert_dropout_output = self.dropout_layer(bert_output)

        final_output = self.dense_layer_4(bert_dropout_output)

        return final_output


class SimilarityTreeLSTM(nn.Block):
    def __init__(self, dropout_rate=0.5):
        super(SimilarityTreeLSTM, self).__init__()
        with self.name_scope():
            self.childsumtreelstm = ChildSumLSTMCell(256, input_size=768)
            #             self.similarity = Similarity(sim_hidden_size, rnn_hidden_size, num_classes)
            self.dropout_layer = nn.Dropout(rate=dropout_rate)
            # self.dense_layer_1 = nn.Dense(128)
            self.dense_layer_2 = nn.Dense(64)
            self.dense_layer_3 = nn.Dense(16)
            self.dense_layer_4 = nn.Dense(2)

    def forward(self, F, token_ids, segment_ids, valid_length, l_sentences, l_trees, r_sentences, r_trees):
        l_output = self.childsumtreelstm(F, l_sentences, l_trees)

        l_dropout_output = self.dropout_layer(l_output)

        l_dense_output = self.dense_layer_2(l_dropout_output)

        r_output = self.childsumtreelstm(F, r_sentences, r_trees)

        r_dropout_output = self.dropout_layer(r_output)

        r_dense_output = self.dense_layer_2(r_dropout_output)

        lstm_concat = F.concat(l_dense_output, r_dense_output, dim=1)

        lstm_concat_dense_output = self.dense_layer_3(lstm_concat)

        lstm_concat_dropout_output = self.dropout_layer(lstm_concat_dense_output)

        final_output = self.dense_layer_4(lstm_concat_dropout_output)

        return final_output
