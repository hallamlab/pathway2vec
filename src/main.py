__author__ = "Abdurrahman Abul-Basher"
__date__ = '15/01/2019'
__copyright__ = "Copyright 2019, The Hallam Lab"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Abdurrahman Abul-Basher"
__email__ = "arbasher@student.ubc.ca"
__status__ = "Production"
__description__ = "This file is the main entry to perform learning and prediction on dataset using path2vec model."

import datetime
import json
import os
import textwrap
from argparse import ArgumentParser

import utility.file_path as fph
from train import train
from utility.arguments import Arguments


def __print_header():
    os.system('clear')
    print('# ' + '=' * 50)
    print('Author: ' + __author__)
    print('Copyright: ' + __copyright__)
    print('License: ' + __license__)
    print('Version: ' + __version__)
    print('Maintainer: ' + __maintainer__)
    print('Email: ' + __email__)
    print('Status: ' + __status__)
    print('Date: ' + datetime.datetime.strptime(__date__, "%d/%m/%Y").strftime("%d-%B-%Y"))
    print('Description: ' + textwrap.TextWrapper(width=45, subsequent_indent='\t     ').fill(__description__))
    print('# ' + '=' * 50)


def save_args(args):
    os.makedirs(args.log_dir, exist_ok=args.exist_ok)
    with open(os.path.join(args.log_dir, 'config.json'), 'wt') as f:
        json.dump(vars(args), f, indent=2)


def __internal_args(parse_args):
    arg = Arguments()

    arg.random_state = parse_args.random_state
    arg.num_jobs = parse_args.num_jobs
    arg.display_interval = parse_args.display_interval
    arg.shuffle = parse_args.shuffle
    arg.num_epochs = parse_args.num_epochs
    arg.max_inner_iter = parse_args.max_inner_iter
    arg.batch = parse_args.batch

    ##########################################################################################################
    ##########                                  ARGUMENTS FOR PATHS                                 ##########
    ##########################################################################################################

    arg.ospath = parse_args.ospath
    arg.inpath = parse_args.inpath
    arg.dspath = parse_args.dspath
    arg.mdpath = parse_args.mdpath
    arg.rspath = parse_args.rspath
    arg.logpath = parse_args.logpath

    ##########################################################################################################
    ##########                          ARGUMENTS FOR FILE NAMES AND MODELS                         ##########
    ##########################################################################################################

    arg.hin_file = parse_args.hin_file
    arg.file_name = parse_args.file_name
    arg.first_graph_name = parse_args.first_graph_name
    arg.second_graph_name = parse_args.second_graph_name
    arg.third_graph_name = parse_args.third_graph_name
    arg.first_mapping_file_name = parse_args.first_mapping_file_name
    arg.second_mapping_file_name = parse_args.second_mapping_file_name
    arg.model_name = parse_args.model_name

    ##########################################################################################################
    ##########                            ARGUMENTS PREPROCESSING FILES                             ##########
    ##########################################################################################################

    arg.preprocess_dataset = parse_args.preprocess_dataset
    arg.extract_instance = parse_args.extract_instance
    arg.include_third_graph = True
    if parse_args.exclude_third_graph:
        arg.include_third_graph = False
    arg.use_metapath_scheme = parse_args.use_metapath_scheme
    arg.metapath_scheme = parse_args.metapath_scheme
    if arg.use_metapath_scheme:
        if parse_args.metapath_scheme is not None or parse_args.metapath_scheme != "":
            arg.metapath_scheme = parse_args.metapath_scheme
        else:
            raise Exception('Please provide a valid metapath scheme.')
    arg.constraint_type = parse_args.constraint_type
    arg.just_type = parse_args.just_type
    arg.just_memory_size = parse_args.just_memory_size
    arg.burn_in_input_size = parse_args.burn_in_input_size
    arg.burn_in_phase = parse_args.burn_in_phase
    arg.walk_length = parse_args.walk_length
    arg.num_walks = parse_args.num_walks
    arg.q = parse_args.q
    arg.first_graph_is_directed = parse_args.first_graph_is_directed
    arg.first_graph_is_connected = parse_args.first_graph_not_connected
    arg.second_graph_is_directed = parse_args.second_graph_is_directed
    arg.second_graph_is_connected = parse_args.second_graph_not_connected
    arg.third_graph_is_directed = parse_args.third_graph_is_directed
    arg.third_graph_is_connected = parse_args.third_graph_not_connected
    arg.remove_isolates = parse_args.remove_isolates
    arg.weighted_within_layers = parse_args.weighted_within_layers

    ##########################################################################################################
    ##########                              ARGUMENTS USED FOR TRAINING                             ##########
    ##########################################################################################################

    arg.train = parse_args.train
    arg.use_truncated_normal_weight = parse_args.use_truncated_normal_weight
    arg.use_truncated_normal_emb = parse_args.use_truncated_normal_emb
    arg.window_size = parse_args.window_size
    arg.num_skips = parse_args.num_skips
    arg.embedding_dim = parse_args.embedding_dim
    arg.negative_samples = parse_args.negative_samples
    arg.learning_rate = parse_args.lr
    arg.subsample_size = parse_args.subsample_size
    arg.max_keep_model = parse_args.max_keep_model
    arg.fit_by_word2vec = parse_args.fit_by_word2vec
    return arg


def parse_command_line():
    __print_header()
    # Parses the arguments.
    parser = ArgumentParser(description="Run path2vec.")

    parser.add_argument('--display-interval', default=-1, type=int,
                        help='display intervals. -1 means display per each iteration.')
    parser.add_argument('--random_state', default=12345, type=int, help='Random seed. (default value: 12345).')
    parser.add_argument('--num-jobs', type=int, default=2, help='Number of parallel workers. (default value: 2).')
    parser.add_argument('--num-epochs', default=3, type=int,
                        help='Number of epochs over the training set. (default value: 3).')
    parser.add_argument('--max-inner-iter', default=1, type=int,
                        help='Number of inner iteration inside a single epoch. '
                             'If batch = 1 better to set to 5. (default value: 1).')
    parser.add_argument('--batch', type=int, default=30, help='Batch size. (default value: 30).')
    parser.add_argument('--shuffle', action='store_false', default=True,
                        help='Whether or not the training data_full should be shuffled after each epoch. (default value: True).')

    # Arguments for path
    parser.add_argument('--ospath', default=fph.OBJECT_PATH, type=str,
                        help='The path to the data_full object that contains extracted '
                             'information from the MetaCyc database. The default is '
                             'set to object folder outside the source code.')
    parser.add_argument('--inpath', default=fph.INPUT_PATH, type=str,
                        help='The path to the input data_full as represented by PathoLogic '
                             'input format. The default is set to inputset folder outside '
                             'the source code.')
    parser.add_argument('--dspath', default=fph.DATASET_PATH, type=str,
                        help='The path to the dataset after the samples are processed. '
                             'The default is set to dataset folder outside the source code.')
    parser.add_argument('--mdpath', default=fph.MODEL_PATH, type=str,
                        help='The path to the output models. The default is set to '
                             'train folder outside the source code.')
    parser.add_argument('--rspath', default=fph.RESULT_PATH, type=str,
                        help='The path to the results. The default is set to result '
                             'folder outside the source code.')
    parser.add_argument('--logpath', default=fph.LOG_PATH, type=str,
                        help='The path to the log directory.')

    # Arguments for file names and models
    parser.add_argument('--hin-file', type=str, default='hin.pkl',
                        help='The file name for the hin object. (default value: "hin.pkl")')
    parser.add_argument('--file-name', type=str, default='hin',
                        help='The file name to save an object. (default value: "hin")')
    parser.add_argument('--first-graph-name', type=str, default='ec_graph.pkl',
                        help='The file name to the first graph. (default value: "ec_graph.pkl")')
    parser.add_argument('--second-graph-name', type=str, default='compound_graph.pkl',
                        help='The file name to the second graph. (default value: "compound_graph.pkl")')
    parser.add_argument('--third-graph-name', type=str, default='pathway_graph.pkl',
                        help='The file name to the third graph. (default value: "pathway_graph.pkl")')
    parser.add_argument('--first-mapping-file-name', type=str, default='ec2compound.pkl',
                        help='The file name to incidence matrix of first to second graphs. '
                             '(default value: "ec2compound.pkl")')
    parser.add_argument('--second-mapping-file-name', type=str, default='compound2pathway.pkl',
                        help='The file name to incidence matrix of second to third graphs. '
                             '(default value: "compound2pathway.pkl")')
    parser.add_argument('--model-name', type=str, default='path2vec',
                        help='The file name, excluding extension, to save an object. (default value: "path2vec")')
    parser.add_argument('--trained-model', type=str, default='path2vec_tf_embeddings.npz',
                        help='The file name, including extension, to save an object. '
                             '(default value: "path2vec_tf_embeddings.npz")')

    # Arguments for preprocessing dataset
    parser.add_argument('--preprocess-dataset', action='store_true', default=False,
                        help='Preprocess graphs to compose a hin. '
                             'Usually, it is applied once.  (default value: False).')
    parser.add_argument('--extract-instance', action='store_true', default=False,
                        help='Extract instances, including metapath based, from a preprocessed multi-graph.')
    parser.add_argument('--exclude-third-graph', action='store_true', default=False,
                        help='Exclude the third graph. (default value: False).')
    parser.add_argument('--metapath-scheme', default='ECTCE', type=str, help='The specified metapath scheme.')
    parser.add_argument('--use-metapath-scheme', action='store_true', default=False,
                        help='Extract instances based on a given metapath pattern. (default value: False).')
    parser.add_argument('--constraint-type', action='store_true', default=False,
                        help='Constraint type. If True, it cares (i.e. heterogeneous negative sampling). '
                             'If False, it does not care (i.e. normal negative sampling). ')
    parser.add_argument('--just-type', action='store_true', default=False,
                        help='Apply jump and stay strategy in random walk. (default value: False).')
    parser.add_argument('--just-memory-size', type=int, default=5,
                        help='The memory size of Q-hist. (default value: 5).')
    parser.add_argument('--burn-in-phase', type=int, default=1,
                        help='Burn in phase time. -1 implies no burn in phase. (default value: 1).')
    parser.add_argument('--burn-in-input-size', default=0.5, type=float,
                        help='Subsampling size of the number of walks and length for burn in phase. (default value: 0.7)')
    parser.add_argument('--num-walks', type=int, default=100, help='Number of walks per source. (default value: 100).')
    parser.add_argument('--walk-length', type=int, default=100, help='Length of walk per source. (default value: 100).')
    parser.add_argument('--q', type=float, default=0.5,
                        help='Explore hyperparameter ([0-1]) denoting the probability of exploring nodes '
                             'at one layer of multi graphs. (default value: 0.5).')
    parser.add_argument('--first-graph-is-directed', action='store_true', default=False,
                        help='Whether the first graph is (un)directed. (default value: undirected).')
    parser.add_argument('--first-graph-not-connected', action='store_false', default=True,
                        help='Whether the first graph is connected. (default value: connected).')
    parser.add_argument('--second-graph-is-directed', action='store_true', default=False,
                        help='Whether the second graph is (un)directed. (default value: undirected).')
    parser.add_argument('--second-graph-not-connected', action='store_false', default=True,
                        help='Whether the second graph is connected. (default value: connected).')
    parser.add_argument('--third-graph-is-directed', action='store_true', default=False,
                        help='Whether the third graph is (un)directed. (default value: undirected).')
    parser.add_argument('--third-graph-not-connected', action='store_false', default=True,
                        help='Whether the third graph is connected. (default value: connected).')
    parser.add_argument('--remove-isolates', action='store_true', default=False,
                        help='Whether the combined multi graphs does not contain isolated nodes. (default value: True).')
    parser.add_argument('--weighted-within-layers', action='store_true', default=False,
                        help='Boolean specifying (un)weighted. (default value: unweighted).')

    # Arguments for training
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train the path2vec model. (default value: False).')
    parser.add_argument('--fit-by-word2vec', action='store_true', default=False,
                        help='Whether to train the path2vec using word2vec. (default value: False).')
    parser.add_argument('--use-truncated-normal-weight', action='store_false', default=True,
                        help='Whether to use truncated normal weightings. (default value: True).')
    parser.add_argument('--use-truncated-normal-emb', action='store_true', default=False,
                        help='Whether to use truncated normal embeddings. (default value: False).')
    parser.add_argument('--window-size', type=int, default=5, help='Context size for optimization. (default value: 5).')
    parser.add_argument('--num-skips', type=int, default=4, help='Number of samples to be considered within defined '
                                                                 'context size. (default value: 4).')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='Dimensionality of the feature vectors. (default value: 128).')
    parser.add_argument('--negative-samples', default=5, type=int,
                        help='Number of negative samples. (default value: 5).')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate. (default value: 0.01).')
    parser.add_argument('--subsample-size', type=int, default=10, help='Subsampling the center nodes. '
                                                                       '-1 implies no subsampling is applied. (default value: 10).')
    parser.add_argument('--max-keep-model', default=100, type=int,
                        help='Number of models to save. (default value: 100).')

    parse_args = parser.parse_args()
    args = __internal_args(parse_args)

    train(arg=args)


if __name__ == "__main__":
    # app.run(parse_command_line)
    parse_command_line()
