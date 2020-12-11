import json
import os
import pickle as pkl
import sys
import traceback

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from scipy.sparse import lil_matrix


def reverse_idx(value2idx):
    idx2value = {}
    for key, value in value2idx.items():
        idx2value.update({value: key})
    return idx2value


def save_sparse(file_name, array):
    np.savez(file_name, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def save_data(data, file_name: str, save_path: str, tag: str = '', mode: str = 'wb', w_string: bool = False,
              print_tag: bool = True):
    '''
    Save data_full into file
    :param mode:
    :param print_tag:
    :param data: the data_full file to be saved
    :param save_path: location of the file
    :param file_name: name of the file
    '''
    try:
        file = file_name
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        file_name = os.path.join(save_path, file_name)
        if print_tag:
            print('\t\t## Storing {0:s} into the file: {1:s}'.format(tag, file))
        with open(file=file_name, mode=mode) as fout:
            if not w_string:
                pkl.dump(data, fout)
            elif w_string:
                fout.write(data)
    except Exception as e:
        print('\t\t## The file {0:s} can not be saved'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def save_json(data, file_name: str, save_path: str, node_link_data: bool = True, tag: str = '', print_tag: bool = True):
    '''
    Save data_full into file
    :param mode:
    :param print_tag:
    :param data: the data_full file to be saved
    :param save_path: location of the file
    :param file_name: name of the file
    '''
    try:
        file = file_name
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        file_name = os.path.join(save_path, file_name)
        if print_tag:
            print('\t\t## Storing {0:s} into the file: {1:s}'.format(tag, file))
        if node_link_data:
            data = json_graph.node_link_data(data)
        with open(file_name, mode='w') as fout:
            json.dump(data, fout, sort_keys=True, indent=4)
    except Exception as e:
        print('\t\t## The file {0:s} can not be saved'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def save_gexf(data, file_name, save_path, tag='', print_tag=True):
    '''
    Save data_full into file
    :param mode:
    :param print_tag:
    :param data: the data_full file to be saved
    :param save_path: location of the file
    :param file_name: name of the file
    '''
    try:
        file = file_name
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        file_name = os.path.join(save_path, file_name)
        if print_tag:
            print('\t\t## Storing {0:s} into the file: {1:s}'.format(tag, file))
        nx.write_gexf(G=data, path=file_name)
    except Exception as e:
        print('\t\t## The file {0:s} can not be saved'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def __load_sparse(file_name):
    while True:
        data = np.load(file_name)
        if type(data) is lil_matrix or type(data) is np.ndarray:
            return data


def load_item_features(file_name, use_components=True):
    try:
        with open(file_name, 'rb') as f_in:
            while True:
                itemFeatures = pkl.load(f_in)
                if use_components:
                    if type(itemFeatures) is tuple:
                        break
                else:
                    if type(itemFeatures) is np.ndarray or type(itemFeatures) is lil_matrix:
                        break
        return itemFeatures
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def load_y_file(num_samples, y_file):
    y_true = list()
    sidx = 0
    try:
        with open(y_file, 'rb') as f_in:
            while sidx < num_samples:
                tmp = pkl.load(f_in)
                if type(tmp) is tuple:
                    y_true.append(tmp[0])
                    sidx += 1
                if sidx == num_samples:
                    break
        return y_true
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(y_file), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def load_y_data(file_name, load_path, tag='selected_pathways sample'):
    try:
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, file_name))
        file_name = os.path.join(load_path, file_name)
        with open(file_name, 'rb') as f_in:
            while True:
                data = pkl.load(f_in)
                if type(data) is tuple and len(data) == 2:
                    y, sample_ids = data
                    break
            return y, sample_ids
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def load_data(file_name, load_path, load_X=False, mode='rb', tag='data_full', print_tag=True):
    '''
    :param data_full:
    :param load_path: load file from a path
    :type file_name: string
    :param file_name:
    '''
    try:
        if print_tag:
            print('\t\t## Loading {0:s} from: {1:s}'.format(tag, file_name))
        file_name = os.path.join(load_path, file_name)
        with open(file_name, mode=mode) as f_in:
            if mode == "r":
                data = f_in.readlines()
            else:
                if load_X:
                    data = __load_sparse(f_in)
                else:
                    data = pkl.load(f_in)
            return data
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def load_json(file_name, load_path, tag='data_full'):
    '''
    Save edus into file
    :param data_full:
    :param load_path: load file from a path
    :type file_name: string
    :param file_name:
    '''
    try:
        print('\t\t## Loading {0:s} from: {1:s}'.format(tag, file_name))
        file_name = os.path.join(load_path, file_name)
        with open(file_name) as f_in:
            data = json.load(f_in)
        return data
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e
