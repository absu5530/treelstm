import models
import dataloader
import mxnet as mx
import gluonnlp
import config
import pandas as pd
import argparse
import spacy
import random
import logging
import warnings
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings('ignore')


class TrainModel(object):
    def __init__(self, args):
        self.args = args
        if self.args.ctx == 'cpu':
            self.ctx = mx.cpu(0)
        elif self.args.ctx == 'gpu':
            self.ctx = mx.gpu(0)

    def get_model(self):
        if self.args.model_type == 'bert+treelstm':
            logging.info("Compiling bert_treelstm model...")
            model = models.BertAndSimilarityTreeLSTM(self.args.dropout)
        elif self.args.model_type == 'bert':
            logging.info("Compiling bert model...")
            model = models.BertClassifier(self.args.dropout)
        elif self.args.model_type == 'treelstm':
            logging.info("Compiling treelstm model...")
            model = models.SimilarityTreeLSTM(self.args.dropout)
        else:
            raise config.BertTreeLSTMException(f'Model {self.args.model_type} not valid.')

        if self.args.mode == 'predict':
            try:
                model.load_parameters(f'./saved_models/{args.experiment_name}_best.params')
                self.model = model
            except Exception as e:
                raise config.BertTreeLSTMException(f'Model params "{args.experiment_name}_best.params" not found: '
                                                   f'{str(e)}')

        else:
            if self.args.model_type in ['bert+treelstm', 'bert']:
                layers_to_freeze = ['transformer' + str(n) + '_' for n in range(0, self.args.no_layers_to_freeze)]
                for p in model.collect_params().values():
                    if any(layer in p.name for layer in layers_to_freeze):
                        p.grad_req = 'null'
                        logging.info(f'Layer {p.name} frozen')
                    else:
                        p.grad_req = 'write'

            if self.args.initializer == 'xavier':
                initializer = mx.init.Xavier(magnitude=2.24)
            elif self.args.initializer == 'normal':
                initializer = mx.init.Normal(0.02)
            else:
                raise config.BertTreeLSTMException("Given initializer invalid.")

            model.collect_params().initialize(init=initializer, ctx=self.ctx)
            self.model = model

    def test(self, data_iter, mode='validationn'):
        logging.info(f'Computing {mode} stats...')

        data_iter.reset()
        metric = mx.metric.Accuracy()
        f1_metric = mx.metric.F1()
        loss_function = mx.gluon.loss.SoftmaxCELoss()
        samples = len(data_iter)
        #     data_iter.set_context(ctx[0])
        #     labels = [mx.nd.array(data_iter.labels, ctx=ctx[0]).reshape((-1,1))]
        l_trees, l_sents, r_trees, r_sents, token_id, segment_id, valid_len, label = data_iter.next(samples)

        token_id = mx.nd.array(token_id).reshape(-1, 128).as_in_context(self.ctx)
        valid_len = mx.nd.array(valid_len).reshape(samples, ).as_in_context(self.ctx).astype('float32')
        segment_id = mx.nd.array(segment_id).reshape(-1, 128).as_in_context(self.ctx)
        label = mx.nd.array(label).as_in_context(self.ctx)

        out = self.model(mx.nd, token_id, segment_id, valid_len, l_sents, l_trees, r_sents, r_trees)
        loss = float(loss_function(out, label).mean().asnumpy())

        #     preds = to_score(mx.nd.concat(*preds, dim=0))
        metric.update(mx.nd.concat(*label, dim=0),
                      mx.nd.concat(*out, dim=0).reshape(-1, 2))

        _, acc = metric.get()

        f1_metric.update(mx.nd.concat(*label, dim=0),
                         mx.nd.concat(*out, dim=0).reshape(-1, 2))

        _, f1 = f1_metric.get()

        cm = confusion_matrix(mx.nd.concat(*label, dim=0).asnumpy(),
                              mx.nd.argmax(mx.nd.concat(*out, dim=0).reshape(-1, 2), axis=1).asnumpy())

        if mode == 'test':
            with open(f'./saved_models/test_results.txt', 'a') as file:
                test_string = f'Experiment {self.args.experiment_name}\n' \
                              f'Accuracy: {round(acc * 100)}%\n' \
                              f'F1: {round(f1, 2)}\n' \
                              f'Loss: {round(loss, 2)}\n' \
                              f'Confusion Matrix: \n{cm}\n'
                logging.info(test_string)
                file.write(test_string)

        return acc, loss

    def random_search(self):
        random.seed(self.args.random_seed)
        self.args.random_seed += 1

        self.args.no_layers_to_freeze = random.choice([3, 7, 10])
        self.args.initializer = random.choice(['xavier', 'normal'])
        self.args.dropout = random.choice([0.2, 0.5, 0.9])
        self.args.batch_size = random.choice([16, 32, 64])
        self.args.learning_rate = random.choice([0.001, 0.0001, 0.00001, 0.000001, 0.0000001])
        self.args.weight_decay = random.choice([self.args.learning_rate / 1000,
                                                self.args.learning_rate / 100,
                                                self.args.learning_rate / 10])

        self.args.experiment_name = self.args.model_type + '_' + \
                                    str(self.args.no_layers_to_freeze) + '_' + \
                                    self.args.initializer + '_' + \
                                    str(self.args.dropout) + '_' + \
                                    str(self.args.batch_size) + '_' + \
                                    str(self.args.learning_rate) + '_' + \
                                    str(self.args.weight_decay)

    def train_model(self):
        experiment_prefix = self.args.experiment_name
        # The hyperparameters
        batch_size = self.args.batch_size
        learning_rate = self.args.learning_rate
        weight_decay = self.args.weight_decay
        num_epochs = self.args.epochs
        early_stopping = self.args.early_stopping

        loss_function = mx.gluon.loss.SoftmaxCELoss()

        logging.info(f'Loading data...')
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe('tensorattr')

        bert_treelstm_dataloader = dataloader.get_data_from_pickle(dataset='train')
        bert_treelstm_dataloader_train, bert_treelstm_dataloader_dev = bert_treelstm_dataloader.reset_and_split(0.8)

        bert_treelstm_dataloader_test = dataloader.get_data_from_pickle(dataset='test')

        train_metric = mx.metric.Accuracy()

        loss_seq_train = []
        loss_seq_val = []
        acc_seq_train = []
        acc_seq_val = []

        trainer = mx.gluon.Trainer(self.model.collect_params(),
                                   'adam',
                                   {'learning_rate': learning_rate,
                                    'wd': weight_decay})

        # Collect all differentiable parameters
        # `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
        # The gradients for these params are clipped later
        params = [p for p in self.model.collect_params().values() if p.grad_req != 'null']

        no_of_batches = int(
            (len(bert_treelstm_dataloader_train) / batch_size) + (len(bert_treelstm_dataloader_train) % batch_size > 0))

        # Training the model with only three epochs
        best_acc = -1
        best_epoch = 0
        for epoch_id in range(num_epochs):
            logging.info(f'Running experiment {experiment_prefix}, epoch {epoch_id}/{num_epochs - 1}...')
            bert_treelstm_dataloader_train.reset()
            #     batch_no = 0
            train_metric.reset()
            epoch_cumulative_log = {'outs': [],
                                    'labels': [],
                                    'losses': []}

            for batch_id in range(no_of_batches):
                logging.info(f'Training for epoch {epoch_id}, batch {batch_id}/{no_of_batches - 1}...')
                l_trees, l_sents, r_trees, r_sents, token_id, segment_id, valid_len, label = bert_treelstm_dataloader_train.next(
                    batch_size)
                n_count = len(valid_len)
                with mx.autograd.record():
                    # Load the data to the CPU/GPU
                    token_id = mx.nd.array(token_id).reshape(-1, 128).as_in_context(self.ctx)
                    valid_len = mx.nd.array(valid_len).reshape(n_count, ).as_in_context(self.ctx).astype('float32')
                    segment_id = mx.nd.array(segment_id).reshape(-1, 128).as_in_context(self.ctx)
                    label = mx.nd.array(label).as_in_context(self.ctx)

                    # Forward computation
                    out = self.model(mx.nd,
                                     token_id,
                                     segment_id,
                                     valid_len.astype('float32'),
                                     l_sents,
                                     l_trees,
                                     r_sents,
                                     r_trees)
                    ls = loss_function(out, label).mean()
                    # And backwards computation
                    ls.backward()

                trainer.allreduce_grads()
                gluonnlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1)

                # Log predictions, labels, loss and batch size for this batch
                epoch_cumulative_log['outs'].extend(out)
                epoch_cumulative_log['labels'].extend(label)
                epoch_cumulative_log['losses'].extend([float(ls.asnumpy())])

            train_metric.update(mx.nd.concat(*epoch_cumulative_log['labels'], dim=0),
                                mx.nd.concat(*epoch_cumulative_log['outs'], dim=0).reshape(-1, 2))
            _, train_acc = train_metric.get()

            train_loss = (sum(epoch_cumulative_log['losses']) / no_of_batches)

            validation_acc, validation_loss = self.test(bert_treelstm_dataloader_dev, mode='validation')

            # Printing vital information
            logging.info(f'Epoch {epoch_id} train acc: {train_acc}')
            logging.info(f'Epoch {epoch_id} train loss: {train_loss}')
            logging.info(f'Epoch {epoch_id} validation acc: {validation_acc}')
            logging.info(f'Epoch {epoch_id} validation loss: {validation_loss}')

            loss_seq_train.append(train_loss)
            loss_seq_val.append(validation_loss)
            acc_seq_train.append(train_acc)
            acc_seq_val.append(validation_acc)

            if validation_acc > best_acc:
                best_acc = validation_acc
                best_epoch = epoch_id
                logging.info('New optimum found for validation acc: {}.'.format(best_acc))
                self.model.save_parameters(f'./saved_models/{experiment_prefix}_best.params')
            else:
                if epoch_id - best_epoch >= early_stopping:
                    logging.info(f'Early stopping at epoch {epoch_id}')
                    break

        losses_and_acc = pd.DataFrame()
        losses_and_acc['Train Loss'] = loss_seq_train
        losses_and_acc['Val Loss'] = loss_seq_val
        losses_and_acc['Train Acc'] = acc_seq_train
        losses_and_acc['Val Acc'] = acc_seq_val
        losses_and_acc.to_csv(f'./saved_models/{experiment_prefix}_losses_and_acc.csv')

        if self.args.mode == 'train-randomsearch':
            with open(f'./saved_models/randomsearch_results.txt', 'a') as file:
                validation_string = f'Experiment name: {experiment_prefix}\n' \
                                    f'Validation accuracy: {int(best_acc * 100)}%\n' \
                                    f'Epoch: {epoch_id}\n'

                logging.info(validation_string)
                file.write(validation_string)
        else:
            _, _ = self.test(bert_treelstm_dataloader_test, mode='test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TODO")
    parser.add_argument("--experiment-name",
                        type=str,
                        default='bert+treelstm',
                        help="Experiment name")
    parser.add_argument("--random-seed",
                        type=int,
                        default=1,
                        help="Random seed for train random search")
    parser.add_argument("--mode",
                        type=str,
                        default='train',
                        choices=['train-randomsearch', 'train', 'predict'],
                        help="Model type")
    parser.add_argument("--permutations",
                        type=int,
                        default=10,
                        help="Number of permutations for random search")
    parser.add_argument("--epochs",
                        type=int,
                        default=30,
                        help="Number of epochs")
    parser.add_argument("--early-stopping",
                        type=int,
                        default=5,
                        help="Number of epochs after which to apply early stopping if no improvement")
    parser.add_argument("--model-type",
                        type=str,
                        default='treelstm',
                        choices=['bert+treelstm', 'bert', 'treelstm'],
                        help="Model type")
    parser.add_argument("--no-layers-to-freeze",
                        type=int,
                        default=7,
                        choices=[3, 7, 10],
                        help="Number of BERT layers to freeze")
    parser.add_argument("--initializer",
                        type=str,
                        default='xavier',
                        choices=['xavier', 'normal'],
                        help="Number of BERT layers to freeze")
    parser.add_argument("--ctx",
                        type=str,
                        default='cpu',
                        choices=['cpu', 'gpu'],
                        help="Experiment name")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.000001,
                        help="Learning rate")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=0.0001,
                        help="Weight decay")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
                        choices=[0.2, 0.5, 0.9],
                        help="Dropout rate")
    # parser.add_argument("--data-path",
    #                     type=str,
    #                     default='./datasets/mrpc/dev.tsv',
    #                     help="Data path for prediction")
    args = parser.parse_args()

    train_obj = TrainModel(args)

    if args.mode == 'train-randomsearch':
        for i in range(0, args.permutations):
            train_obj.random_search()
            train_obj.get_model()
            train_obj.train_model()
    elif args.mode == 'predict':
        train_obj.get_model()
        test_iter = dataloader.get_data_from_pickle(dataset='test')
        train_obj.test(test_iter, mode='test')
    elif args.mode == 'train':
        train_obj.get_model()
        train_obj.train_model()
    else:
        raise config.BertTreeLSTMException(f'Invalid mode given.')
    # if not args.predict:
    # else:
    #     print('predict')
    #     predict_data = dataloader.get_data_from_scratch(dataset=train_obj.args.data_path)
    #     _, _, _, out, _ = train_obj.test(predict_data)
    #     output_results = f'Accuracy: {int(acc * 100)}%\n' \
    #                      f'F1: {round(f1, 2)}\n' \
    #                      f'Loss: {round(loss, 2)}\n'
    #     print(output_results)

    # else:
    #     for i in range(0, args.n_runs):
    #         #             prefix = "set_" + str(args.set_no) + "_run_" + str(
    #         #                 i) + "_nf_" + str(args.n_filters) + "_fs1_" + str(
    #         #                 args.filter_size_1) + "_fs2_" + str(
    #         #                 args.filter_size_2) + "_fs3_" + str(
    #         #                 args.filter_size_3) + "_dp_" + str(
    #         #                 args.dropout_rate) + '_2fc_str_' + str(
    #         #                 args.strides) + '_ksize_' + str(args.ksize) + '_lp_' + str(
    #         #                 args.learning_power) + '_pos'
    #         prefix = "experiment_{}_{}_{}".format(args.experiment, args.set_no, args.variable)
    #         train_obj.test(args, prefix)
