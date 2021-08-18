import random
random.seed(100)
import gluonnlp
import data_transform
import pickle
import config
import spacy
from spacy.language import Language
import numpy as np


class BertTreeLSTMIter(object):
    """
    Iterator class for MRPC BERTTreeLSTM dataset. Adapted from:
    https://gluon.mxnet.io/chapter09_natural-language-processing/tree-lstm.html
    """
    def __init__(self, num_classes, shuffle=False):
        super(BertTreeLSTMIter, self).__init__()
        self.l_sentences = []
        self.r_sentences = []
        self.l_trees = []
        self.r_trees = []
        self.token_ids = []
        self.segment_ids = []
        self.valid_length = []
        self.labels = []
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            mask = list(range(len(self.l_sentences)))
            random.shuffle(mask)
            self.l_sentences = [self.l_sentences[i] for i in mask]
            self.r_sentences = [self.r_sentences[i] for i in mask]
            self.l_trees = [self.l_trees[i] for i in mask]
            self.r_trees = [self.r_trees[i] for i in mask]
            self.token_ids = [self.token_ids[i] for i in mask]
            self.segment_ids = [self.segment_ids[i] for i in mask]
            self.valid_length = [self.valid_length[i] for i in mask]
            self.labels = [self.labels[i] for i in mask]
        self.index = 0

    def reset_and_split(self, train_proportion=0.8):
        self.reset()
        train_size = int(train_proportion * len(self.labels))

        dataloader_train = BertTreeLSTMIter(self.num_classes)
        dataloader_train.token_ids = self.token_ids[:train_size]
        dataloader_train.segment_ids = self.segment_ids[:train_size]
        dataloader_train.valid_length = self.valid_length[:train_size]
        dataloader_train.l_sentences = self.l_sentences[:train_size]
        dataloader_train.r_sentences = self.r_sentences[:train_size]
        dataloader_train.l_trees = self.l_trees[:train_size]
        dataloader_train.r_trees = self.r_trees[:train_size]
        dataloader_train.labels = self.labels[:train_size]
        dataloader_train.index = 0

        dataloader_dev = BertTreeLSTMIter(self.num_classes)
        dataloader_dev.token_ids = self.token_ids[train_size:]
        dataloader_dev.segment_ids = self.segment_ids[train_size:]
        dataloader_dev.valid_length = self.valid_length[train_size:]
        dataloader_dev.l_sentences = self.l_sentences[train_size:]
        dataloader_dev.r_sentences = self.r_sentences[train_size:]
        dataloader_dev.l_trees = self.l_trees[train_size:]
        dataloader_dev.r_trees = self.r_trees[train_size:]
        dataloader_dev.labels = self.labels[train_size:]
        dataloader_dev.index = 0
        return dataloader_train, dataloader_dev

    def next(self, batch_size=1):
        out = self[self.index:self.index + batch_size]
        self.index += batch_size
        return out

    def set_context(self, context):
        self.l_sentences = [a.as_in_context(context) for a in self.l_sentences]
        self.r_sentences = [a.as_in_context(context) for a in self.r_sentences]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        l_tree = self.l_trees[index]
        r_tree = self.r_trees[index]
        l_sent = self.l_sentences[index]
        r_sent = self.r_sentences[index]
        token_id = self.token_ids[index]
        segment_id = self.segment_ids[index]
        valid_len = self.valid_length[index]
        label = self.labels[index]
        return l_tree, l_sent, r_tree, r_sent, token_id, segment_id, valid_len, label


@Language.factory('tensorattr')
class TensorAttr:
    """
    # Custom class to expose BERT tensors in spaCy pipeline. Adapted from:
    https://applied-language-technology.readthedocs.io/en/latest/notebooks/part_iii/04_embeddings_continued.html
    """
    # We continue by defining the first method of the class,
    # __init__(), which is called when this class is used for
    # creating a Python object. Custom components in spaCy
    # require passing two variables to the __init__() method:
    # 'name' and 'nlp'. The variable 'self' refers to any
    # object created using this class!
    def __init__(self, name, nlp):
        # We do not really do anything with this class, so we
        # simply move on using 'pass' when the object is created.
        pass

    # The __call__() method is called whenever some other object
    # is passed to an object representing this class. Since we know
    # that the class is a part of the spaCy pipeline, we already know
    # that it will receive Doc objects from the preceding layers.
    # We use the variable 'doc' to refer to any object received.
    def __call__(self, doc):
        # When an object is received, the class will instantly pass
        # the object forward to the 'add_attributes' method. The
        # reference to self informs Python that the method belongs
        # to this class.
        self.add_attributes(doc)

        # After the 'add_attributes' method finishes, the __call__
        # method returns the object.
        return doc

    # Next, we define the 'add_attributes' method that will modify
    # the incoming Doc object by calling a series of methods.
    def add_attributes(self, doc):
        #         # spaCy Doc objects have an attribute named 'user_hooks',
        #         # which allows customising the default attributes of a
        #         # Doc object, such as 'vector'. We use the 'user_hooks'
        #         # attribute to replace the attribute 'vector' with the
        #         # Transformer output, which is retrieved using the
        #         # 'doc_tensor' method defined below.
        doc.user_hooks['vector'] = self.doc_tensor

        #         # We then perform the same for both Spans and Tokens that
        #         # are contained within the Doc object.
        #         doc.user_span_hooks['vector'] = self.span_tensor
        doc.user_token_hooks['vector'] = self.token_tensor

    #         # We also replace the 'similarity' method, because the
    #         # default 'similarity' method looks at the default 'vector'
    #         # attribute, which is empty! We must first replace the
    #         # vectors using the 'user_hooks' attribute.
    #         doc.user_hooks['similarity'] = self.get_similarity
    #         doc.user_span_hooks['similarity'] = self.get_similarity
    #         doc.user_token_hooks['similarity'] = self.get_similarity

    # Define a method that takes a Doc object as input and returns
    # Transformer output for the entire Doc.
    def doc_tensor(self, doc):
        # Return Transformer output for the entire Doc. As noted
        # above, this is the last item under the attribute 'tensor'.
        # Average the output along axis 0 to handle batched outputs.
        return doc._.trf_data.tensors[-1].mean(axis=0)

    # Define a method that takes a Span as input and returns the Transformer
    # output.
    def span_tensor(self, span):
        # Get alignment information for Span. This is achieved by using
        # the 'doc' attribute of Span that refers to the Doc that contains
        # this Span. We then use the 'start' and 'end' attributes of a Span
        # to retrieve the alignment information. Finally, we flatten the
        # resulting array to use it for indexing.
        tensor_ix = span.doc._.trf_data.align[span.start: span.end].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 ("columns"). This yields a 768-dimensional
        # vector for each spaCy Span.
        return tensor.mean(axis=0)

    # Define a function that takes a Token as input and returns the Transformer
    # output.
    def token_tensor(self, token):
        # Get alignment information for Token; flatten array for indexing.
        # Again, we use the 'doc' attribute of a Token to get the parent Doc,
        # which contains the Transformer output.
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()

        # Fetch Transformer output shape from the final dimension of the output.
        # We do this here to maintain compatibility with different Transformers,
        # which may output tensors of different shape.
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]

        # Get Token tensors under tensors[0]. Reshape batched outputs so that
        # each "row" in the matrix corresponds to a single token. This is needed
        # for matching alignment information under 'tensor_ix' to the Transformer
        # output.
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]

        # Average vectors along axis 0 (columns). This yields a 768-dimensional
        # vector for each spaCy Token.
        return tensor.mean(axis=0)

    # Define a function for calculating cosine similarity between vectors
    def get_similarity(self, doc1, doc2):
        # Calculate and return cosine similarity
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)


def get_raw_data(dataset='train'):
    """
    Get raw MRPC data from where it is saved.
    """
    if dataset == 'train':
        data_raw = gluonnlp.data.GlueMRPC('train', root='./datasets/mrpc')
    elif dataset == 'test':
        data_raw = gluonnlp.data.GlueMRPC('dev', root='./datasets/mrpc')
    else:
        raise config.BertTreeLSTMException(f'Invalid  dataset {dataset}.')
    return data_raw


def bert_transform_data(data_raw):
    """
    Get BERT model and transform raw data to BERT classifier format. Adapted from GluonNLP documentation:
    https://nlp.gluon.ai/examples/sentence_embedding/bert.html
    """
    bert_base, vocabulary = gluonnlp.model.get_model('bert_12_768_12',
                                                     dataset_name='book_corpus_wiki_en_uncased',
                                                     pretrained=True, use_pooler=True,
                                                     use_decoder=False, use_classifier=False)

    sample_id = 0
    # Sentence A
    print(data_raw[sample_id][0])
    # Sentence B
    print(data_raw[sample_id][1])
    # 1 means equivalent, 0 means not equivalent
    print(data_raw[sample_id][2])

    # Use the vocabulary from pre-trained model for tokenization
    bert_tokenizer = gluonnlp.data.BERTTokenizer(vocabulary, lower=True)

    # The maximum length of an input sequence
    max_len = 128

    # The labels for the two classes [(0 = not similar) or  (1 = similar)]
    all_labels = ["0", "1"]

    # whether to transform the data as sentence pairs.
    # for single sentence classification, set pair=False
    # for regression task, set class_labels=None
    # for inference without label available, set has_label=False
    pair = True
    transform = data_transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                    class_labels=all_labels,
                                                    has_label=True,
                                                    pad=True,
                                                    pair=pair)
    data_transformed = data_raw.transform(transform)

    print('vocabulary used for tokenization = \n%s' % vocabulary)
    print('%s token id = %s' % (vocabulary.padding_token, vocabulary[vocabulary.padding_token]))
    print('%s token id = %s' % (vocabulary.cls_token, vocabulary[vocabulary.cls_token]))
    print('%s token id = %s' % (vocabulary.sep_token, vocabulary[vocabulary.sep_token]))
    print('token ids = \n%s' % data_transformed[sample_id][0])
    print('segment ids = \n%s' % data_transformed[sample_id][1])
    print('valid length = \n%s' % data_transformed[sample_id][2])
    print('label = \n%s' % data_transformed[sample_id][3])

    return data_transformed


def get_spacy_trees(data_raw):
    """
    Get spaCy documents and dependency trees for every sentence. `_left` indicates the data object for the
    left sentence and `_right` for the right sentence in the sentence pair.
    """
    nlp = spacy.load("en_core_web_trf")

    nlp.add_pipe('tensorattr')

    # Call the 'pipeline' attribute to examine the pipeline
    print('spaCy pipeline:')
    print(nlp.pipeline)

    data_train_nlp_left = list(nlp.pipe([sample[0] for sample in data_raw]))
    data_train_nlp_right = list(nlp.pipe([sample[1] for sample in data_raw]))

    data_train_nlp_trees_left = []
    for nlp_text in data_train_nlp_left:
        sentence = list(nlp_text.sents)[0]
        data_train_nlp_trees_left.append(sentence.root)

    data_train_nlp_trees_right = []
    for nlp_text in data_train_nlp_right:
        sentence = list(nlp_text.sents)[0]
        data_train_nlp_trees_right.append(sentence.root)

    return data_train_nlp_left, data_train_nlp_right, data_train_nlp_trees_left, data_train_nlp_trees_right


def get_data_from_scratch(dataset='train'):
    """
    Get raw data and process it for the model. This involves getting token_ids, segment_ids and lengths for the BERT
    classifier and spaCy doc objects (sentences) and dependency trees for the left sentence and the right sentence
    in each sentence pair.
    """
    data_raw = get_raw_data(dataset=dataset)
    data_transformed = bert_transform_data(data_raw)

    with open(f'./data/data_transformed_{dataset}.pickle', 'wb') as handle:
        pickle.dump(data_transformed, handle)

    data_train_nlp_left, \
    data_train_nlp_right, \
    data_train_nlp_trees_left, \
    data_train_nlp_trees_right = get_spacy_trees(data_raw)

    with open(f'./data/data_train_nlp_left_{dataset}.pickle', 'wb') as handle:
        pickle.dump(data_train_nlp_left, handle)

    with open(f'./data/data_train_nlp_right_{dataset}.pickle', 'wb') as handle:
        pickle.dump(data_train_nlp_right, handle)

    bert_treelstm_dataloader = BertTreeLSTMIter(2)
    bert_treelstm_dataloader.token_ids = [d[0] for d in data_transformed]
    bert_treelstm_dataloader.segment_ids = [d[1] for d in data_transformed]
    bert_treelstm_dataloader.valid_length = [d[2] for d in data_transformed]
    bert_treelstm_dataloader.l_sentences = data_train_nlp_left
    bert_treelstm_dataloader.r_sentences = data_train_nlp_right
    bert_treelstm_dataloader.l_trees = data_train_nlp_trees_left
    bert_treelstm_dataloader.r_trees = data_train_nlp_trees_right
    bert_treelstm_dataloader.labels = [d[3] for d in data_transformed]

    return bert_treelstm_dataloader


def get_data_from_pickle(dataset='train'):
    """
    Retrieve processed data for modeling, including token_ids, segment_ids and lengths for the BERT
    classifier and spaCy doc objects (sentences) and dependency trees for the left sentence and the right sentence
    in each sentence pair.
    """
    try:
        with open(f'./data/data_transformed_{dataset}.pickle', 'rb') as handle:
            data_transformed = pickle.load(handle)

        with open(f'./data/data_train_nlp_left_{dataset}.pickle', 'rb') as handle:
            data_train_nlp_left = pickle.load(handle)

        with open(f'./data/data_train_nlp_right_{dataset}.pickle', 'rb') as handle:
            data_train_nlp_right = pickle.load(handle)
    except EOFError:
        raise config.BertTreeLSTMException("EOF Error at get_data_from_pickle. Pickle file is possibly empty. "
                                           "Get data from scratch first to recreate pickle files.")

    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe('tensorattr')

    data_train_nlp_trees_left = []
    for nlp_text in data_train_nlp_left:
        sentence = list(nlp_text.sents)[0]
        data_train_nlp_trees_left.append(sentence.root)

    data_train_nlp_trees_right = []
    for nlp_text in data_train_nlp_right:
        sentence = list(nlp_text.sents)[0]
        data_train_nlp_trees_right.append(sentence.root)

    bert_treelstm_dataloader = BertTreeLSTMIter(2)
    bert_treelstm_dataloader.token_ids = [d[0] for d in data_transformed]
    bert_treelstm_dataloader.segment_ids = [d[1] for d in data_transformed]
    bert_treelstm_dataloader.valid_length = [d[2] for d in data_transformed]
    bert_treelstm_dataloader.l_sentences = data_train_nlp_left
    bert_treelstm_dataloader.r_sentences = data_train_nlp_right
    bert_treelstm_dataloader.l_trees = data_train_nlp_trees_left
    bert_treelstm_dataloader.r_trees = data_train_nlp_trees_right
    bert_treelstm_dataloader.labels = [d[3] for d in data_transformed]

    return bert_treelstm_dataloader


if __name__ == "__main__":
    get_data_from_scratch(dataset='train')
    get_data_from_scratch(dataset='test')
