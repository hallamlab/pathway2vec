'''
This file is used to preprocess the input dataset, and
train it using pathway2vec model.
'''

import sys
import time
import traceback

from hin import MetaPathGraph
from model.path2vec import path2vec
from utility.access_file import load_data


def __train(arg):
    '''
    Create training objData by calling the Data class
    '''

    # Setup the number of operations to employ
    steps = 1

    ##########################################################################################################
    #########################       PREPROCESS GRAPHS AND GENERTE RANDOM WALKS       #########################
    ##########################################################################################################

    # Whether to display parameters at every operation
    display_params = True

    if arg.preprocess_dataset:
        print('\n{0})- Preprocess graph used for training and evaluating...'.format(steps))
        steps = steps + 1
        model = MetaPathGraph(first_graph_is_directed=arg.first_graph_is_directed,
                              first_graph_is_connected=arg.first_graph_is_connected,
                              second_graph_is_directed=arg.second_graph_is_directed,
                              second_graph_is_connected=arg.second_graph_is_connected,
                              third_graph_is_directed=arg.third_graph_is_directed,
                              third_graph_is_connected=arg.third_graph_is_connected,
                              weighted_within_layers=arg.weighted_within_layers,
                              remove_isolates=arg.remove_isolates, q=arg.q,
                              num_walks=arg.num_walks, walk_length=arg.walk_length,
                              learning_rate=arg.learning_rate, num_jobs=arg.num_jobs,
                              display_interval=arg.display_interval)
        print('\t>> Loading graphs and association matrices...')
        first_graph = load_data(file_name=arg.first_graph_name, load_path=arg.ospath, tag='first graph')
        second_graph = load_data(file_name=arg.second_graph_name, load_path=arg.ospath,
                                 tag='second graph')
        first_mapping_file = load_data(file_name=arg.first_mapping_file_name, load_path=arg.ospath,
                                       tag='association matrix of first to second graphs')
        third_graph = None
        second_mapping_file = None
        if arg.include_third_graph:
            third_graph = load_data(file_name=arg.third_graph_name, load_path=arg.ospath,
                                    tag='third graph')
            second_mapping_file = load_data(file_name=arg.second_mapping_file_name, load_path=arg.ospath,
                                            tag='association matrix of second to  third graphs')
        model.parse_graph_to_hin(first_graph=first_graph, second_graph=second_graph, third_graph=third_graph,
                                 first_mapping_file=first_mapping_file, second_mapping_file=second_mapping_file,
                                 hin_file=arg.hin_file, ospath=arg.ospath, display_params=display_params)
        display_params = False

    if arg.extract_instance:
        print('\n{0})- Extract walks...'.format(steps))
        steps = steps + 1
        print('\t>> Loading the heterogeneous information network...')
        hin = load_data(file_name=arg.hin_file, load_path=arg.ospath,
                        tag='heterogeneous information network')
        model = MetaPathGraph(first_graph_is_directed=arg.first_graph_is_directed,
                              first_graph_is_connected=arg.first_graph_is_connected,
                              second_graph_is_directed=arg.second_graph_is_directed,
                              second_graph_is_connected=arg.second_graph_is_connected,
                              third_graph_is_directed=arg.third_graph_is_directed,
                              third_graph_is_connected=arg.third_graph_is_connected,
                              weighted_within_layers=arg.weighted_within_layers,
                              remove_isolates=arg.remove_isolates, q=arg.q,
                              num_walks=arg.num_walks, walk_length=arg.walk_length,
                              learning_rate=arg.learning_rate, num_jobs=arg.num_jobs,
                              display_interval=arg.display_interval)
        model.generate_walks(constraint_type=arg.constraint_type, just_type=arg.just_type,
                             just_memory_size=arg.just_memory_size, use_metapath_scheme=arg.use_metapath_scheme,
                             metapath_scheme=arg.metapath_scheme, burn_in_phase=arg.burn_in_phase,
                             burn_in_input_size=arg.burn_in_input_size, hin=hin,
                             save_file_name=arg.file_name, ospath=arg.ospath, dspath=arg.dspath,
                             display_params=display_params)

    ##########################################################################################################
    ######################                  PATHWAY2VEC MODEL                  ###############################
    ##########################################################################################################

    # Whether to display parameters at every operation
    display_params = True

    if arg.train:
        print('\n{0})- Training dataset using pathway2vec model...'.format(steps))
        print('\t>> Loading the heterogeneous information network and training samples...')
        hin = load_data(file_name=arg.hin_file, load_path=arg.ospath,
                        tag='heterogeneous information network')
        X = load_data(file_name=arg.file_name, load_path=arg.dspath, mode='r', tag='dataset')
        X = [sample.strip().split('\t') for sample in X]
        index2type = dict((val, key) for key, list_val in hin.type2index.items() for val in list_val)
        model = path2vec(node_size=hin.number_of_nodes(), window_size=arg.window_size, num_skips=arg.num_skips,
                         num_negative_samples=arg.negative_samples, embedding_dimension=arg.embedding_dim,
                         use_truncated_normal_weight=arg.use_truncated_normal_weight,
                         use_truncated_normal_emb=arg.use_truncated_normal_emb, constraint_type=arg.constraint_type,
                         learning_rate=arg.learning_rate, num_models=arg.max_keep_model,
                         subsample_size=arg.subsample_size, batch=arg.batch, num_epochs=arg.num_epochs,
                         max_inner_iter=arg.max_inner_iter, num_jobs=arg.num_jobs, shuffle=arg.shuffle,
                         display_interval=arg.display_interval, random_state=arg.random_state)
        node_probability = dict((node[1]['mapped_idx'], node[1]['weight'].data)
                                for node in hin.nodes(data=True))
        model.fit(X=X, node_probability=node_probability, index2type=index2type, type2index=hin.type2index,
                  type2prob=hin.type2prob, fit_by_word2vec=arg.fit_by_word2vec, model_name=arg.model_name,
                  model_path=arg.mdpath, result_path=arg.rspath, display_params=display_params)


def train(arg):
    try:
        if arg.preprocess_dataset or arg.extract_instance or arg.train:
            actions = list()
            if arg.preprocess_dataset:
                actions += ['PREPROCESS BIOCYC GRAPHS']
            if arg.extract_instance:
                actions += ['GENERATE RANDOM WALKS']
            desc = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(actions))), actions)]
            desc = ' '.join(desc)
            print('\n*** APPLIED ACTIONS ARE: {0}'.format(desc))
            timeref = time.time()
            __train(arg)
            print('\n*** The selected actions consumed {1:f} SECONDS\n'.format('', round(time.time() - timeref, 3)),
                  file=sys.stderr)
        else:
            print('\n*** PLEASE SPECIFY AN ACTION...\n', file=sys.stderr)
    except Exception:
        print(traceback.print_exc())
        raise
