'''
This file is used to train the input dataset.
'''

import collections
import logging
import os
import sys
import time
import warnings

import numpy as np
import tensorflow as tf
from gensim.models import word2vec
from model.emb import Embedding
from tensorflow.contrib.tensorboard.plugins import projector
from utility.access_file import save_data

logger = logging.getLogger(__name__)
EPSILON = np.finfo(np.float32).eps


class path2vec(object):
    def __init__(self, node_size=10, window_size=5, num_skips=1, num_negative_samples=5, embedding_dimension=100,
                 use_truncated_normal_weight=True, use_truncated_normal_emb=False, constraint_type=True,
                 learning_rate=0.01, num_models=100, subsample_size=10, batch=100, num_epochs=5, max_inner_iter=500,
                 num_jobs=-1, shuffle=True, display_interval=-1, random_state=12345, log_path='../../log'):
        logging.basicConfig(filename=os.path.join(log_path, 'path2vec_Events'), level=logging.DEBUG)
        tf.set_random_seed(seed=random_state)
        self.node_size = node_size
        self.window_size = window_size
        self.num_skips = num_skips
        self.num_negative_samples = num_negative_samples
        self.embedding_dimension = embedding_dimension
        self.use_truncated_normal_weight = use_truncated_normal_weight
        self.use_truncated_normal_emb = use_truncated_normal_emb
        self.learning_rate = learning_rate
        self.constraint_type = constraint_type
        self.display_interval = display_interval
        self.num_models = num_models
        self.batch = batch
        self.subsample_size = subsample_size
        self.num_epochs = num_epochs
        self.max_inner_iter = max_inner_iter
        self.num_jobs = num_jobs
        self.shuffle = shuffle
        self.random_state = random_state
        self.log_path = log_path
        self.params = list()
        warnings.filterwarnings("ignore", category=Warning)

    def __print_arguments(self, **kwargs):
        argdict = dict()
        argdict.update({'node_size': 'Number of nodes: {0}'.format(self.node_size)})
        argdict.update({'window_size': 'Context size: {0}'.format(self.window_size)})
        argdict.update(
            {'num_skips': 'Number of samples to be considered within defined context size: {0}'.format(self.num_skips)})
        argdict.update({'num_negative_samples': 'Number of negative samples: {0}'.format(self.num_negative_samples)})
        argdict.update(
            {'embedding_dimension': 'Dimensionality of the feature vectors: {0}'.format(self.embedding_dimension)})
        argdict.update({'use_truncated_normal_weight': 'Use truncated normal weightings? {0}'.format(
            self.use_truncated_normal_weight)})
        argdict.update(
            {'use_truncated_normal_emb': 'Use truncated normal embeddings? {0}'.format(self.use_truncated_normal_emb)})
        argdict.update({'constraint_type': 'Heterogeneous negative sampling? {0}'.format(self.constraint_type)})
        argdict.update({'learning_rate': 'The learning rate: {0}'.format(self.learning_rate)})
        argdict.update(
            {'max_inner_iter': 'Number of inner iterations within a single epoch: {0}'.format(self.max_inner_iter)})
        argdict.update({'num_epochs': 'Number of epochs over training set: {0}'.format(self.num_epochs)})
        argdict.update({'batch': 'Batch size: {0}'.format(self.batch)})
        argdict.update({'subsampling_size': 'Subsampling size: {0}'.format(self.subsample_size)})
        argdict.update({'display_interval': 'Display Interval: {0}'.format(self.display_interval)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})
        argdict.update({'num_models': 'Number of models to keep: {0}'.format(self.num_models)})
        argdict.update({'shuffle': 'Shuffle the datset? {0}'.format(self.shuffle)})
        argdict.update({'log_path': 'The location of the log information : {0}'.format(self.log_path)})
        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)
        logger.info('\t>> The following arguments are applied:\n\t\t{0}'.format(args))

    def __shffule(self, X):
        if self.shuffle:
            np.random.shuffle(X)

    def __optimizer(self, center_node, context_node, negative_samples):
        # Extract the embeddings
        embed_matrix = tf.trainable_variables()[0]

        # Perform lookup for embeddings based on the inputs
        ct_lookup = tf.nn.embedding_lookup(embed_matrix, center_node)
        ct_expand = tf.tile(ct_lookup, tf.stack([1, self.num_negative_samples, 1]))
        ct_expand.set_shape([None, ct_expand.get_shape()[1], ct_expand.get_shape()[2]])
        ctx_lookup = tf.nn.embedding_lookup(embed_matrix, context_node)
        neg_lookup = tf.nn.embedding_lookup(embed_matrix, negative_samples)
        batch_size = tf.shape(ct_lookup)[0]

        # Dot products between context and center nodes
        ct_ctx = tf.diag_part(tf.tensordot(ct_lookup, ctx_lookup, axes=[[1, 2], [1, 2]]))
        # Dot products between negative candidates and center nodes
        ct_neg = tf.diag_part(tf.tensordot(ct_expand, neg_lookup, axes=[[2], [2]]))

        # Gradients of context nodes
        sigmoid_ct_ctx = tf.sigmoid(ct_ctx - 1)
        grad_context = tf.squeeze(tf.tensordot(tf.diag(sigmoid_ct_ctx), ct_lookup, axes=1))

        # Gradients of negative nodes
        sigmoid_ct_neg = tf.reshape(tf.sigmoid(ct_neg), [-1])
        sigmoid_ct_neg = tf.reshape(sigmoid_ct_neg, [tf.shape(sigmoid_ct_neg)[0], 1])
        ct_expand = tf.reshape(ct_expand, [batch_size * self.num_negative_samples, -1])
        grad_negative = tf.multiply(sigmoid_ct_neg, ct_expand)

        # Gradients of center nodes
        grad_center_1 = tf.squeeze(tf.tensordot(tf.diag(sigmoid_ct_ctx), ctx_lookup, axes=1))
        grad_center_2 = tf.multiply(sigmoid_ct_neg, tf.reshape(neg_lookup,
                                                               [batch_size * self.num_negative_samples,
                                                                tf.shape(neg_lookup)[2]]))
        grad_center_2 = tf.reshape(grad_center_2, [batch_size, self.num_negative_samples, tf.shape(neg_lookup)[2]])
        grad_center_2 = tf.reduce_sum(grad_center_2, axis=1)
        grad_center = grad_center_1 + grad_center_2

        # Multiply all gradients with prespecified learning rate
        all_grads = tf.concat([grad_center, grad_context, grad_negative], axis=0)
        all_grads = tf.multiply(self.learning_rate, all_grads)

        # Calculate expected mean by nodes indices
        neg_reshape = tf.reshape(negative_samples, [batch_size * self.num_negative_samples, 1])
        batch_nodes_idx = tf.reshape(tf.concat([center_node, context_node, neg_reshape], axis=0), [-1])
        unique_batch_nodes_idx = tf.unique(batch_nodes_idx)[0]
        num_unique_batch_nodes_idx = tf.shape(unique_batch_nodes_idx)[0]
        num_features = tf.shape(all_grads)[1]
        idx = tf.constant(0)
        all_mean = tf.zeros(shape=[1, ], dtype=tf.float32)
        cond = lambda idx, all_mean: idx < num_unique_batch_nodes_idx

        def body(idx, all_mean):
            node_idx = unique_batch_nodes_idx[idx]
            node_idx_reshaped = tf.reshape(node_idx, [1, -1])
            num_rows = tf.shape(batch_nodes_idx)[0]
            partitions = tf.cast(tf.reduce_any(tf.equal(tf.reshape(batch_nodes_idx, [-1, 1]), node_idx_reshaped), 1),
                                 tf.int32)
            rows_to_gather = tf.dynamic_partition(tf.range(num_rows), partitions, 2)[1]
            slice_values = tf.gather(all_grads, rows_to_gather)
            expected_mean = tf.reduce_mean(slice_values, axis=0)
            expected_mean = embed_matrix[node_idx] - expected_mean
            new_mean = tf.cond(tf.equal(idx, 0), lambda: expected_mean,
                               lambda: tf.concat([all_mean, expected_mean], axis=0))
            return idx + 1, new_mean

        # Iterate to all over nodes
        _, expected_grad = tf.while_loop(cond=cond, body=body, loop_vars=(idx, all_mean),
                                         shape_invariants=(idx.shape, tf.TensorShape([None])))
        expected_grad = tf.reshape(expected_grad, shape=[num_unique_batch_nodes_idx, num_features])

        # Update embeddings
        embed_matrix = tf.scatter_update(embed_matrix, unique_batch_nodes_idx, expected_grad)

        return embed_matrix

    def __build_tf_place_holders(self, node_probability=None):
        tf.reset_default_graph()

        with tf.name_scope('inputs'):
            center_node = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='center_node')
            context_node = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='context_node')
            negative_samples = tf.placeholder(dtype=tf.int32, shape=[None, self.num_negative_samples],
                                              name='negative_samples')
        with tf.name_scope('embeddings'):
            emb = Embedding(node_size=self.node_size, embedding_dimension=self.embedding_dimension,
                            use_truncated_normal_embedding=self.use_truncated_normal_emb)
            embed_matrix = tf.Variable(emb.E, dtype=tf.float32, name='embedding_matrix')

        with tf.name_scope('probability'):
            if node_probability is None:
                node_probability = 1 / self.node_size * np.ones(shape=(self.node_size, 1), dtype=np.float32)
            node_probability = node_probability.reshape((self.node_size, 1))
            prob_node = tf.Variable(node_probability, dtype=tf.float32, name='node_probability')

        ct_lookup = tf.nn.embedding_lookup(embed_matrix, center_node, name='embedding_lookup_center')
        ctx_lookup = tf.nn.embedding_lookup(embed_matrix, context_node, name='embedding_lookup_context')
        neg_lookup = tf.nn.embedding_lookup(embed_matrix, negative_samples, name='embedding_lookup_negative')
        prob_lookup = tf.nn.embedding_lookup(prob_node, negative_samples)
        prob_lookup = tf.reshape(prob_lookup, (self.batch, self.num_negative_samples))

        with tf.name_scope('loss'):
            # Define loss function to be negative sampling function
            ct_ctx = tf.diag_part(tf.tensordot(ct_lookup, ctx_lookup, axes=[[1, 2], [1, 2]]))
            ct_lookup_expand = tf.tile(ct_lookup, tf.stack([1, self.num_negative_samples, 1]))
            ct_lookup_expand.set_shape([None, ct_lookup_expand.get_shape()[1], ct_lookup_expand.get_shape()[2]])
            ct_neg = tf.log_sigmoid(tf.diag_part(-tf.tensordot(ct_lookup_expand, neg_lookup, axes=[[2], [2]])))
            expected_neg = tf.multiply(prob_lookup, ct_neg)
            loss = tf.reduce_mean(tf.reduce_sum(tf.log_sigmoid(ct_ctx)) + tf.reduce_sum(expected_neg), name='loss')

        tf.losses.add_loss(tf.reduce_mean(embed_matrix ** 2) * 1e-6)
        tf.summary.scalar(name='loss_summary', tensor=loss)

        return center_node, context_node, negative_samples, loss

    def __generate_batch(self, X):
        assert self.num_skips <= self.window_size * 2
        span = 2 * self.window_size + 1
        padding_string = '-1'
        padding = self.window_size * [padding_string]
        batch_size = self.batch * self.subsample_size
        buffer = collections.deque(maxlen=span)
        list_X = list()
        for sample in X:
            sample = padding + sample + padding
            buffer.extend(sample[0:0 + span])
            sample = sample[0 + span:]
            for item in sample:
                context_nodes_idx = [w for w in range(span) if w != self.window_size]
                context_nodes_idx = [idx for idx in context_nodes_idx if buffer[idx] != padding_string]
                num_skips = self.num_skips
                if len(context_nodes_idx) < self.num_skips:
                    num_skips = self.num_skips
                context_idx_to_use = np.random.choice(a=context_nodes_idx, size=num_skips, replace=False)
                tmp = [(buffer[self.window_size], buffer[idx]) for idx in context_idx_to_use if
                       buffer[self.window_size] != buffer[idx]]
                list_X.extend(tmp)
                buffer.append(item)
        list_X = list(set(list_X))
        self.__shffule(X=list_X)
        indices = np.arange(len(list_X))
        if self.subsample_size != -1:
            if batch_size < len(list_X):
                indices = np.random.choice(a=np.arange(len(list_X)), size=batch_size, replace=False)
        list_X = np.array(list_X, dtype=np.int32)
        center_node_batch = list_X[indices][:, 0].reshape((indices.shape[0], 1))
        context_node_batch = list_X[indices][:, 1].reshape((indices.shape[0], 1))
        return center_node_batch, context_node_batch

    def __get_negative_samples(self, center_nodes, node_id, node_probability, index2type, type2index, type2probs):
        # If constraint_type is set to True then it's based on heterogeneous negative sampling,
        # which represents metapath based training
        if not self.constraint_type:
            negative_samples = [
                np.random.choice(node_id, size=self.num_negative_samples, replace=False, p=node_probability)
                for idx in np.arange(center_nodes.shape[0])]
        else:
            negative_samples = list()
            for node in center_nodes:
                node_type = index2type[int(node)]
                sampling_probs = type2probs[node_type].toarray()
                sampling_probs = np.reshape(sampling_probs, newshape=(sampling_probs.shape[0],))
                sampling_candidates = np.array(type2index[node_type])
                negative_samples_indices = np.random.choice(len(sampling_candidates), size=self.num_negative_samples,
                                                            replace=False, p=sampling_probs)
                negative_samples.append(sampling_candidates[negative_samples_indices])
        negative_samples = np.array(negative_samples, dtype=np.int32).reshape(
            (len(negative_samples), self.num_negative_samples))
        return negative_samples

    def __fit_by_word2vec(self, X, type2index, model_name, model_path, result_path):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        old_cost = np.inf
        timeref = time.time()
        cost_file_name = model_name + "_word2vec_cost.txt"
        save_data('', file_name=cost_file_name, save_path=result_path, mode='w', w_string=True, print_tag=False)
        print('\t>> Training by word2vec model...')
        logger.info('\t>> Training by word2vec model...')
        model = word2vec.Word2Vec(size=self.embedding_dimension, window=self.window_size, min_count=0,
                                  sg=1, workers=self.num_jobs, negative=self.num_negative_samples,
                                  compute_loss=True)
        print('\t>> Building vocabulary...')
        logger.info('\t>> Building vocabulary...')
        model.build_vocab(X)
        n_epochs = self.num_epochs + 1
        if self.constraint_type:
            n_epochs = self.num_epochs + 2
            node_type = [t for t, nodes in type2index.items()]
            list_type = list()
            for items, t in enumerate(node_type):
                list_type.append([str(node) for node in type2index[t] if str(node) in model])
        for epoch in np.arange(start=1, stop=n_epochs):
            desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(epoch, n_epochs - 1)
            print(desc)
            logger.info(desc)
            self.__shffule(X=X)
            list_batches = np.arange(start=0, stop=len(X), step=self.batch)
            epoch_timeref = time.time()
            new_cost = 0.0
            for idx, batch in enumerate(list_batches):
                desc = '\t       --> Learning: {0:.2f}% ...'.format(((idx + 1) / len(list_batches)) * 100)
                logger.info(desc)
                if (idx + 1) != len(list_batches):
                    print(desc, end="\r")
                if (idx + 1) == len(list_batches):
                    print(desc)
                model.train(X[batch:batch + self.batch], total_examples=len(X[batch:batch + self.batch]),
                            epochs=self.max_inner_iter, compute_loss=True)
                if self.constraint_type:
                    for items in list_type:
                        emb = model[items]
                        denominator = np.sum(np.triu(np.dot(emb, emb.T), 1))
                        emb = emb / denominator
                        for i, node in enumerate(items):
                            model.wv.syn0[model.wv.vocab[node].index] = emb[i]
                new_cost += model.get_latest_training_loss() / len(list_batches)
                new_cost /= self.max_inner_iter
            if self.constraint_type and epoch == 1:
                continue
            self.is_fit = True
            print('\t\t  ## Epoch {0} took {1} seconds...'.format(epoch, round(time.time() - epoch_timeref, 3)))
            logger.info(
                '\t\t  ## Epoch {0} took {1} seconds...'.format(epoch, round(time.time() - epoch_timeref, 3)))
            data = str(epoch) + '\t' + str(round(time.time() - epoch_timeref, 3)) + '\t' + str(new_cost) + '\n'
            save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                      print_tag=False)
            # Save models parameters based on test frequencies
            if (epoch % self.display_interval) == 0 or epoch == 1 or epoch == n_epochs - 1:
                print('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                logger.info('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))

                if new_cost < old_cost or epoch == n_epochs - 1:
                    old_cost = new_cost

                    tag_final_file = "_word2vec.ckpt"
                    tag_final_embeddings = "_word2vec_embeddings.npz"
                    if epoch == n_epochs - 1:
                        tag_final_file = "_final_word2vec.ckpt"
                        tag_final_embeddings = "_final_word2vec_embeddings.npz"

                    print('\t\t  --> Storing the path2vec model to: {0:s}'.format(model_name + tag_final_file))
                    logger.info('\t\t  --> Storing the path2vec model to: {0:s}'.format(model_name + tag_final_file))
                    model.wv.save_word2vec_format(os.path.join(model_path, model_name + tag_final_file))

                    print('\t\t  --> Storing the path2vec node embeddings as numpy array to: {0:s}'.format(
                        model_name + tag_final_embeddings))
                    logger.info('\t\t  --> Storing the path2vec node embeddings as numpy array to: {0:s}'.format(
                        model_name + tag_final_embeddings))
                    model_embeddings = np.zeros((self.node_size, self.embedding_dimension), dtype=np.float32)
                    for v_idx in np.arange(self.node_size):
                        if str(v_idx) in model.wv.vocab:
                            model_embeddings[v_idx] = model[str(v_idx)]
                    np.savez(os.path.join(model_path, model_name + tag_final_embeddings), model_embeddings)

        print('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
        logger.info('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))

    def __fit_by_tf(self, X, node_id, node_probability, index2type, type2index, type2prob, model_name, model_path,
                    result_path):
        ## Build layers for path2vec
        print('\t>> Building: path2vec layers...')
        logger.info('\t>> Building: path2vec layers...')
        timeref = time.time()
        center_node_holder, context_node_holder, negative_samples_holder, loss = self.__build_tf_place_holders(
            node_probability=node_probability)
        ## Optimization function for path2vec
        optimizer = self.__optimizer(center_node_holder, context_node_holder, negative_samples_holder)

        print('\t\t## Building layers consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
        logger.info('\t\t## Building layers consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
        print('\t>> Training path2vec...')
        logger.info('\t>> Training path2vec...')
        old_cost = np.inf
        timeref = time.time()
        cost_file_name = model_name + "_cost.txt"
        save_data('', file_name=cost_file_name, save_path=result_path, mode='w', w_string=True, print_tag=False)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=self.num_models)
        config = tf.ConfigProto(intra_op_parallelism_threads=0,
                                inter_op_parallelism_threads=0,
                                allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(self.log_path, sess.graph)
            # Define metadata variable.
            run_metadata = tf.RunMetadata()
            for epoch in np.arange(start=1, stop=self.num_epochs + 1):
                desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(epoch, self.num_epochs)
                print(desc)
                logger.info(desc)
                self.__shffule(X=X)
                list_batches = np.arange(start=0, stop=len(X), step=self.batch)
                epoch_timeref = time.time()
                new_cost = 0.0
                for idx, batch in enumerate(list_batches):
                    total_samples = (idx + 1) / len(list_batches)
                    desc = '\t       --> Learning: {0:.4f}% ...'.format(total_samples * 100)
                    logger.info(desc)
                    if (idx + 1) != len(list_batches):
                        print(desc, end="\r")
                    if (idx + 1) == len(list_batches):
                        print(desc)
                    ## Generate batch negative samples
                    center_nodes, context_nodes = self.__generate_batch(X=X[batch:batch + self.batch])
                    negative_nodes = self.__get_negative_samples(center_nodes=center_nodes, node_id=node_id,
                                                                 node_probability=node_probability,
                                                                 index2type=index2type, type2index=type2index,
                                                                 type2probs=type2prob)
                    batch_X_size = self.batch
                    if self.batch > 150000:
                        batch_X_size = 10000

                    list_batch_X = np.arange(start=0, stop=center_nodes.shape[0], step=batch_X_size)
                    for b_idx, batch_X_idx in enumerate(list_batch_X):
                        center_batch = center_nodes[batch_X_idx:batch_X_idx + batch_X_size]
                        context_batch = context_nodes[batch_X_idx:batch_X_idx + batch_X_size]
                        negative_batch = negative_nodes[batch_X_idx:batch_X_idx + batch_X_size]
                        for inner_iterations in np.arange(self.max_inner_iter):
                            feed_dict = {center_node_holder: center_batch,
                                         context_node_holder: context_batch,
                                         negative_samples_holder: negative_batch}
                            # We perform one update step by evaluating the optimizer op (including it
                            # in the list of returned values for session.run()
                            # Also, evaluate the merged op to get all summaries from the returned
                            # "summary" variable. Feed metadata variable to session for visualizing
                            # the graph in TensorBoard.
                            loss_batch, _, summary_str = sess.run([loss, optimizer, merged],
                                                                  feed_dict=feed_dict,
                                                                  run_metadata=run_metadata)
                            writer.add_summary(summary_str, inner_iterations)
                            loss_batch /= center_batch.shape[0]
                            new_cost += loss_batch / self.max_inner_iter
                    new_cost /= len(list_batch_X)
                new_cost /= len(list_batches)
                new_cost = new_cost * -1
                self.is_fit = True
                print('\t\t  ## Epoch {0} took {1} seconds...'.format(epoch, round(time.time() - epoch_timeref, 3)))
                logger.info(
                    '\t\t  ## Epoch {0} took {1} seconds...'.format(epoch, round(time.time() - epoch_timeref, 3)))
                data = str(epoch) + '\t' + str(round(time.time() - epoch_timeref, 3)) + '\t' + str(new_cost) + '\n'
                save_data(data=data, file_name=cost_file_name, save_path=result_path, mode='a', w_string=True,
                          print_tag=False)
                # Save models parameters based on test frequencies
                if (epoch % self.display_interval) == 0 or epoch == 1 or epoch == self.num_epochs:
                    print('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                    logger.info('\t\t  --> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                    if new_cost < old_cost or epoch == self.num_epochs:
                        old_cost = new_cost
                        tag_final_file = "_tf.ckpt"
                        tag_final_embeddings = "_tf_embeddings.npz"
                        if epoch == self.num_epochs:
                            tag_final_file = "_final_tf.ckpt"
                            tag_final_embeddings = "_final_tf_embeddings.npz"

                        print('\t\t  --> Storing the path2vec model to: {0:s}'.format(model_name + tag_final_file))
                        logger.info(
                            '\t\t  --> Storing the path2vec model to: {0:s}'.format(model_name + tag_final_file))
                        saver.save(sess, os.path.join(model_path, model_name + tag_final_file))

                        print('\t\t  --> Storing the path2vec node embeddings as numpy array to: {0:s}'.format(
                            model_name + tag_final_embeddings))
                        logger.info('\t\t  --> Storing the path2vec node embeddings as numpy array to: {0:s}'.format(
                            model_name + tag_final_embeddings))
                        model_embeddings = tf.get_default_graph()
                        model_embeddings = model_embeddings.get_tensor_by_name("embeddings/embedding_matrix:0")
                        # Create a configuration for visualizing embeddings with the selected_pathways in TensorBoard.
                        # TODO: comment this
                        config = projector.ProjectorConfig()
                        embedding_conf = config.embeddings.add()
                        embedding_conf.tensor_name = model_embeddings.name
                        ##
                        model_embeddings = sess.run(model_embeddings)
                        np.savez(os.path.join(model_path, model_name + tag_final_embeddings), model_embeddings)
                        # TODO: comment this
                        embedding_conf.metadata_path = os.path.join(model_path, model_name + '_metadata.tsv')
                        projector.visualize_embeddings(writer, config)
            writer.close()
            print('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
            logger.info('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))

    def fit(self, X, node_probability=None, index2type=None, type2index=None, type2prob=None,
            fit_by_word2vec: bool = False, model_name='path2vec', model_path="../../model",
            result_path=".", display_params: bool = True):
        if display_params:
            self.__print_arguments(
                fit_by_word2vec='Fit by word2vec: {0}'.format(fit_by_word2vec),
                model_path='The location of the trained model: {0}'.format(model_path),
                result_path='The location of the results (epoch, time, cost): {0}'.format(result_path))
            time.sleep(2)

        if fit_by_word2vec:
            self.__fit_by_word2vec(X=X, type2index=type2index, model_name=model_name, model_path=model_path,
                                   result_path=result_path)
        else:
            if X is None:
                raise Exception("Please provide a dataset.")
            node_id = np.array([key for key, val in node_probability.items()])
            node_probability = np.array([val for key, val in node_probability.items()])
            tmp = np.zeros(shape=(len(node_probability),))
            for idx, val in enumerate(node_probability):
                if not val:
                    continue
                tmp[idx] = val[0]
            node_probability = tmp
            self.__fit_by_tf(X=X, node_id=node_id, node_probability=node_probability, index2type=index2type,
                             type2index=type2index, type2prob=type2prob, model_name=model_name, model_path=model_path,
                             result_path=result_path)
