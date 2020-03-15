'''
This file is used to preprocess the input dataset that is
represented as graphs and to generate random walks.
'''

import collections
import logging
import networkx as nx
import numpy as np
import os
import sys
import time
import warnings
from multiprocessing import Pool
from scipy.sparse import lil_matrix
from utility.access_file import save_data

EPSILON = 0.0001

logger = logging.getLogger(__name__)


class MetaPathGraph(object):
    def __init__(self, first_graph_is_directed: bool = False, first_graph_is_connected: bool = True,
                 second_graph_is_directed: bool = False, second_graph_is_connected: bool = True,
                 third_graph_is_directed: bool = False, third_graph_is_connected: bool = True,
                 weighted_within_layers: bool = False, remove_isolates: bool = True, q: float = 1.0,
                 num_walks: int = 100, walk_length: int = 100, learning_rate: float = 0.001, 
                 num_jobs: int = 1, display_interval: int = 50, log_path='../../log'):
        logging.basicConfig(filename=os.path.join(log_path, 'MetaPathGraph_Events'), level=logging.DEBUG)
        self.first_graph_is_directed = first_graph_is_directed
        self.first_graph_is_connected = first_graph_is_connected
        self.second_graph_is_directed = second_graph_is_directed
        self.second_graph_is_connected = second_graph_is_connected
        self.third_graph_is_directed = third_graph_is_directed
        self.third_graph_is_connected = third_graph_is_connected
        self.weighted_within_layers = weighted_within_layers
        self.remove_isolates = remove_isolates
        self.q = np.sqrt(q)
        self.p = np.sqrt(1 - q)
        self.learning_rate = learning_rate
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.num_jobs = num_jobs
        self.display_interval = display_interval
        warnings.filterwarnings("ignore", category=Warning)

    def __print_arguments(self, **kwargs):
        argdict = dict()
        argdict.update({'first_graph_is_directed': 'The nodes in the first graph are directed? {0}'.format(
            self.first_graph_is_directed)})
        argdict.update({'first_graph_is_connected': 'The nodes in the first graph are connected? {0}'.format(
            self.first_graph_is_connected)})
        argdict.update({'second_graph_is_directed': 'The nodes in the second graph are directed? {0}'.format(
            self.second_graph_is_directed)})
        argdict.update({'second_graph_is_connected': 'The nodes in the second graph are connected? {0}'.format(
            self.second_graph_is_connected)})
        argdict.update({'third_graph_is_directed': 'The nodes in the third graph are directed? {0}'.format(
            self.third_graph_is_directed)})
        argdict.update({'third_graph_is_connected': 'The nodes in the third graph are connected? {0}'.format(
            self.third_graph_is_connected)})
        argdict.update({'weighted_within_layers': 'Edges within graphs in a multi-graphs are weighted? {0}'.format(
            self.weighted_within_layers)})
        argdict.update(
            {'remove_isolates': 'Remove isolated nodes from a multi-graphs? {0}'.format(self.remove_isolates)})
        argdict.update({
            'p': 'Return parameter that controls the likelihood of immediately revisiting a node in the walk: {0}'.format(
                self.p)})
        argdict.update({
            'q': 'In-out parameter that allows the search to differentiate between "inward" and "outward" nodes: {0}'.format(
                self.q)})
        argdict.update({'learning_rate': 'The learning rate: {0}'.format(self.learning_rate)})
        argdict.update({'num_walks': 'Number of walks per source node: {0}'.format(self.num_walks)})
        argdict.update({'walk_length': 'Length of walk per source node: {0}'.format(self.walk_length)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})
        argdict.update({'display_interval': 'Display Interval: {0}'.format(self.display_interval)})
        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)
        logger.info('\t>> The following arguments are applied:\n\t\t{0}'.format(args))

    def __alpha(self, next_node, next_node_type=None, prev_node=None, neighbours_prev_node=None, curr_node_type=None,
                weight_curr_node=1.0, weight_curr_node_type=1.0, weight_next_node_type=1.0, explore_layer=False,
                constraint_type=False):
        if constraint_type:
            if explore_layer:
                if curr_node_type == next_node_type:
                    prob = self.q * weight_curr_node_type
                else:
                    prob = self.p * weight_next_node_type
            else:
                prob = [self.q * weight_curr_node, self.p * weight_curr_node]
                prob = prob / np.sum(prob)
                i = np.random.choice(a=[0, 1], p=prob)
                prob = prob[i]
            return prob
        else:
            if next_node in neighbours_prev_node:
                return 0.5
            elif prev_node == next_node:
                return self.p
            else:
                return self.q

    def __check_metapath_validity(self, metapath_scheme):
        '''
        Check errors associated with the metapath guideline
        :param metapath_scheme:
        '''
        error = False
        desc = ''
        if metapath_scheme[0:2] != metapath_scheme[3:][::-1]:
            desc += '\n\t   --> Error: the metapath is incorrectly specified...'
            logger.info(desc)
            error = True
        if metapath_scheme[2] == metapath_scheme[0] or metapath_scheme[2] == metapath_scheme[1]:
            desc += '\n\t   --> Error: the metapath is incorrectly specified...'
            logger.info(desc)
            error = True
        if len(metapath_scheme) - 1 > self.walk_length:
            desc += '\n\t   --> The number of walks is lower than the metapath scheme...'
            logger.info(desc)
            self.walk_length = len(metapath_scheme) - 1
        if error:
            raise Exception(desc)

    def __compose_graphs(self, first_graph, second_graph, first_adjaceny_matrix, third_graph=None,
                         second_adjaceny_matrix=None):
        if self.first_graph_is_directed:
            first_graph = nx.Graph(first_graph).to_directed()
        if self.second_graph_is_directed:
            second_graph = nx.Graph(second_graph).to_directed()
        if third_graph is not None:
            if self.third_graph_is_directed:
                third_graph = nx.Graph(third_graph).to_directed()
        if self.weighted_within_layers:
            ebc = nx.edge_betweenness_centrality(first_graph, normalized=True)
            nx.set_edge_attributes(first_graph, values=ebc, name='weight')
            ebc = nx.edge_betweenness_centrality(second_graph, normalized=True)
            nx.set_edge_attributes(second_graph, values=ebc, name='weight')
            if third_graph is not None:
                ebc = nx.edge_betweenness_centrality(third_graph, normalized=True)
                nx.set_edge_attributes(third_graph, values=ebc, name='weight')
        else:
            nx.set_edge_attributes(first_graph, values=1, name='weight')
            nx.set_edge_attributes(second_graph, values=1, name='weight')
            if third_graph is not None:
                nx.set_edge_attributes(third_graph, values=1, name='weight')

        hin = nx.MultiGraph()
        hin.add_nodes_from(first_graph.nodes(data=True))
        if self.first_graph_is_connected:
            hin.add_weighted_edges_from(first_graph.edges(data='weight'))
        hin.add_nodes_from(second_graph.nodes(data=True))
        if self.second_graph_is_connected:
            hin.add_weighted_edges_from(second_graph.edges(data='weight'))
        if third_graph is not None:
            hin.add_nodes_from(third_graph.nodes(data=True))
            if self.third_graph_is_connected:
                hin.add_weighted_edges_from(third_graph.edges(data='weight'))
        second_tmp = dict(second_graph.nodes(data='idx'))
        if third_graph is not None:
            third_tmp = dict(third_graph.nodes(data='idx'))

        count = 1
        for e_id, e_data in first_graph.nodes(data=True):
            desc = '\t   --> Processing: {0:.4f}%'.format(count / len(first_graph.nodes) * 100)
            logger.info(desc)
            if count % self.display_interval == 0:
                print(desc, end="\r")
            if count == len(first_graph.nodes):
                print(desc)
            count = count + 1
            e_idx = int(e_data['idx'])
            c_list = np.nonzero(first_adjaceny_matrix[e_idx, :])[1]
            for c_idx in c_list:
                c_id = list(second_tmp.keys())[list(second_tmp.values()).index(c_idx)]
                if not hin.has_edge(e_id, c_id):
                    hin.add_edges_from([(e_id, c_id, {'weight': 1})])
                if third_graph is not None:
                    p_list = np.nonzero(second_adjaceny_matrix[c_idx, :])[1]
                    for p_idx in p_list:
                        p_id = list(third_tmp.keys())[list(third_tmp.values()).index(p_idx)]
                        if not hin.has_edge(c_id, p_id):
                            hin.add_edges_from([(c_id, p_id, {'weight': 1})])
        return hin

    def parse_graph_to_hin(self, first_graph, second_graph, third_graph=None, first_mapping_file='ec2compound.pkl',
                           second_mapping_file=None, hin_file='hin.pkl', ospath='objectset',
                           display_params: bool = True):
        if display_params:
            self.__print_arguments()
            time.sleep(2)
        print('\t>> Building a multi-modal graph...')
        logger.info('\t>> Building a multi-modal graph...')
        hin = self.__compose_graphs(first_graph=first_graph, second_graph=second_graph,
                                    first_adjaceny_matrix=first_mapping_file, third_graph=third_graph,
                                    second_adjaceny_matrix=second_mapping_file)
        if self.remove_isolates:
            print('\t\t--> Removing {0:d} isolated nodes from the multi-modal graph...'.format(
                len(list(nx.isolates(hin)))))
            logger.info('\t\t--> Removing {0:d} isolated nodes from the multi-modal graph...'.format(
                len(list(nx.isolates(hin)))))
            hin.remove_nodes_from(list(nx.isolates(hin)))

        save_data(data=hin, file_name=hin_file, save_path=ospath, tag='heterogeneous information network',
                  mode='w+b')

    def __init_probability(self, hin):
        print('\t>> Map nodes type to index and estimate nodes initial stationary distribution...')
        logger.info('\t>> Map nodes type to index and estimate nodes initial stationary distribution...')
        nodes2triplets = dict((node[0], (i, node[1]['idx'], node[1]['type']))
                              for i, node in enumerate(hin.nodes(data=True)))
        set_types = set([node_data for node_id, node_data in hin.nodes(data='type')])
        type2index = dict((node_type, []) for node_type in set_types)
        count = 1
        init_node_prob = lil_matrix((len(nodes2triplets), 1))
        for node in nodes2triplets.items():
            desc = '\t\t--> Processing {0:.4f}%'.format(count / len(nodes2triplets) * 100)
            logger.info(desc)
            if count != len(nodes2triplets):
                print(desc, end="\r")
            if count == len(nodes2triplets):
                print(desc)
            count = count + 1
            neighbors = list(hin.neighbors(node[0]))
            attrs = {node[0]: {'mapped_idx': node[1][0]}}
            nx.set_node_attributes(hin, attrs)
            type2index[node[1][2]].append(node[1][0])
            init_node_prob[node[1][0]] = len(neighbors) + 1
        init_node_prob = init_node_prob.power(3.0 / 4.0)
        type2prob = dict()
        for node_type in set_types:
            indicies_for_a_type = type2index[node_type]
            type2prob[node_type] = init_node_prob[indicies_for_a_type]
            tmp = type2prob[node_type].sum()
            type2prob[node_type] = lil_matrix(type2prob[node_type].multiply(1 / tmp))
        tmp = init_node_prob.sum()
        init_node_prob = init_node_prob.multiply(1 / tmp)
        return init_node_prob, type2index, type2prob

    def _walks_per_node(self, node_idx, node_curr, node_curr_data, hin, just_memory_size, trans_prob, dspath=".",
                        save_file_name=".", burn_in_phase=False):
        if len(list(hin.neighbors(node_curr))) == 0:
            desc = '\t\t\t--> Extracted walks for {0:.4f}% of nodes...'.format(
                ((node_idx + 1) / hin.number_of_nodes()) * 100)
            print(desc, end="\r")
            if burn_in_phase:
                return trans_prob
            else:
                return
        if hin.trans_metapath_scheme:
            metapath_scheme = None
            if node_curr_data['type'] in hin.metapath_scheme:
                frequent_scheme = hin.metapath_scheme[: -1] * 2
                idx = str(frequent_scheme).index(node_curr_data['type'])
                metapath_scheme = frequent_scheme[idx: idx + len(hin.metapath_scheme) - 1]
                metapath_scheme = metapath_scheme * (self.walk_length // len(metapath_scheme))
            if metapath_scheme is None:
                if burn_in_phase:
                    return trans_prob
                else:
                    return
            if node_curr_data['type'] != metapath_scheme[0]:
                desc = '\t\t\t--> Extracted walks for {0:.4f}% of nodes...'.format(
                    ((node_idx + 1) / hin.number_of_nodes()) * 100)
                print(desc, end="\r")
                if burn_in_phase:
                    return trans_prob
                else:
                    return
        walk_length = self.walk_length
        num_walks = self.num_walks + 1
        if burn_in_phase:
            num_walks = int(self.num_walks * self.burn_in_input_size)
            walk_length = int(self.walk_length * self.burn_in_input_size)
            if num_walks < 0:
                num_walks = 10
            if walk_length < 0:
                walk_length = 10
        for curr_walk in np.arange(start=1, stop=num_walks):
            X = [node_curr_data['mapped_idx']]
            prev_node = [node_curr]
            curr_node = node_curr
            curr_node_data = node_curr_data
            # The size of memory to hold the nodes types
            q_hist = collections.deque(maxlen=just_memory_size)
            q_hist.extend(node_curr_data['type'])
            for curr_length in np.arange(start=1, stop=walk_length):
                if curr_length > 1:
                    list_neigh_idx_prev_node = [hin.nodes[edge[1]]['mapped_idx'] for edge in
                                                hin.edges(prev_node[-2])]
                    prev_node_idx = X[-2]
                else:
                    list_neigh_idx_prev_node = [hin.nodes[edge[1]]['mapped_idx'] for edge in
                                                hin.edges(prev_node[-1])]
                    prev_node_idx = X[-1]

                if hin.trans_metapath_scheme:
                    neigh_curr_node = [(edge[1], edge[2]) for edge in hin.edges(curr_node, data='weight')
                                       if hin.nodes[edge[1]]['type'] == metapath_scheme[curr_length]]
                    if len(neigh_curr_node) == 0:
                        neigh_curr_node = [(edge[1], edge[2]) for edge in hin.edges(curr_node, data='weight')
                                           if hin.nodes[edge[1]]['type'] == metapath_scheme[curr_length - 1]]
                else:
                    neigh_curr_node = [(edge[1], edge[2]) for edge in hin.edges(curr_node, data='weight')]

                list_neigh_curr_node = np.array([node[0] for node in neigh_curr_node])
                neigh_type_curr_node = np.array([hin.nodes[v]['type'] for v in list_neigh_curr_node])
                neigh_idx_curr_node = np.array([hin.nodes[node]['mapped_idx'] for node in list_neigh_curr_node])

                # Retrieve weights of nodes (usually set to 1.) at the start of burn in phase;
                # otherwise, retrieve the previous transition probabilities.
                trans_from_curr_node = trans_prob[X[-1], neigh_idx_curr_node].toarray()[0]
            
                if hin.trans_constraint_type or hin.trans_just_type:
                    # Compute the transition probability based on types of the current node's neighbours.
                    # We further smooth the transition probabilities by adding 0.001 to weights of current
                    # node, next node and current node type.
                    trans_node_type = [self.__alpha(next_node=hin.nodes[next_node]['mapped_idx'],
                                                    next_node_type=neigh_type_curr_node[idx],
                                                    curr_node_type=curr_node_data['type'],
                                                    weight_curr_node=len(
                                                        list(hin.neighbors(next_node))) + 0.001,
                                                    weight_curr_node_type=sum(
                                                        neigh_type_curr_node == curr_node_data['type']) + 0.001,
                                                    weight_next_node_type=sum(
                                                        neigh_type_curr_node == hin.nodes[next_node][
                                                            'type']) + 0.001,
                                                    explore_layer=hin.trans_just_type, constraint_type=True)
                                       for idx, next_node in enumerate(list_neigh_curr_node)]
                    trans_node_type = np.multiply(trans_node_type, trans_from_curr_node)

                    if hin.trans_just_type and not hin.trans_metapath_scheme:
                        if len(q_hist) == just_memory_size:
                            available_types = set(q_hist)
                            for t in available_types:
                                # Explore within a layer more frequently as suggested by JUST; however,
                                # the JUST algorithm is modified to explore a wider range when the memory
                                # size in Q_hist is larger than the nodes types.
                                # Note, when q == p then we recover the JUST algorithm.
                                if hin.q != hin.p:
                                    weight_decay = 1 / q_hist.count(t)
                                    if q_hist.count(t) == int(just_memory_size * hin.q):
                                        weight_decay = -q_hist.count(t)
                                else:
                                    weight_decay = -q_hist.count(t)
                                tmp = trans_node_type[neigh_type_curr_node == t] * np.exp(weight_decay)
                                trans_node_type[neigh_type_curr_node == t] = tmp
                    trans_node_type = trans_node_type / np.sum(trans_node_type)
                    node_type = np.random.choice(neigh_type_curr_node, size=1, p=trans_node_type)

                    # Include only those nodes that have the same chosen type.
                    list_neigh_curr_node = [(edge[1], edge[2]) for edge in
                                            hin.edges(curr_node, data='weight')
                                            if hin.nodes[edge[1]]['type'] == node_type]
                    neigh_idx_curr_node = np.array(
                        [hin.nodes[node[0]]['mapped_idx'] for node in list_neigh_curr_node])

                    # Retrieve weights of nodes (usually set to 1.) at the start of burn in phase;
                    # otherwise, retrieve the previous transition probabilities.
                    trans_from_curr_node = trans_prob[X[-1], neigh_idx_curr_node].toarray()[0]
                    list_neigh_curr_node = np.array([node[0] for node in list_neigh_curr_node])

                # Compute the transition probability of the current node's neighbours based on the chosen type.
                # We further smooth the transition probabilities by adding 0.001 to weights of current node.
                trans_prob_next_node = [
                    self.__alpha(next_node=hin.nodes[next_node]['mapped_idx'], prev_node=prev_node_idx,
                                 neighbours_prev_node=list_neigh_idx_prev_node,
                                 weight_curr_node=len(list(hin.neighbors(next_node))) + 0.001)
                    for next_node in list_neigh_curr_node]
                trans_prob_next_node = np.multiply(trans_prob_next_node, trans_from_curr_node)
                trans_prob_next_node = trans_prob_next_node / np.sum(trans_prob_next_node)
                next_node = np.random.choice(neigh_idx_curr_node, 1, p=trans_prob_next_node)[0]

                # If the transition probability is not computed then initialize it with the most recent
                # estimation; otherwise update the existing one.
                tmp = trans_prob_next_node[neigh_idx_curr_node == next_node]
                trans_prob[X[-1], next_node] = trans_prob[X[-1], next_node] + tmp * self.learning_rate
                curr_node = list_neigh_curr_node[neigh_idx_curr_node == next_node][0]
                curr_node_data = hin.nodes[curr_node]
                # Store the sequence of simulated walks and nodes in Q hist upto predefined memory size.
                X = X + [next_node]
                prev_node = prev_node + [curr_node]
                q_hist.extend(hin.nodes[curr_node]['type'])
            # Save the generated instances into the .txt file
            if not burn_in_phase:
                X = '\t'.join([str(v) for v in X])
                save_data(data=X + '\n', file_name=save_file_name, save_path=dspath, mode='a', w_string=True,
                        print_tag=False)
                desc = '\t\t\t--> Extracted walks for {0:.4f}% of nodes...'.format(
                    ((node_idx + 1) / hin.number_of_nodes()) * 100)
                print(desc, end="\r")
        if burn_in_phase:
            return trans_prob
            
                

    def generate_walks(self, constraint_type, just_type, just_memory_size,
                       use_metapath_scheme, metapath_scheme='ECTCE', burn_in_phase: int = 10,
                       burn_in_input_size: float = 0.5,
                       hin='hin.pkl', save_file_name='hin', ospath='objectset', 
                       dspath='dataset', display_params: bool = True):
        if burn_in_phase < 0:
            burn_in_phase = 1
        self.burn_in_phase = burn_in_phase
        if burn_in_input_size < 0 or burn_in_input_size > 1:
            burn_in_input_size = 0.1
        self.burn_in_input_size = burn_in_input_size
        if use_metapath_scheme:
            if metapath_scheme.strip() != '' or metapath_scheme is not None:
                self.__check_metapath_validity(metapath_scheme=metapath_scheme)
                hin.metapath_scheme = metapath_scheme
            else:
                desc = '\n\t   --> Error: Please provide a metapath scheme...'
                logger.warning(desc)
                raise Exception(desc)
        else:
            hin.metapath_scheme = None
            metapath_scheme = None

        if display_params:
            self.__print_arguments(use_metapath_scheme='Use a metapath scheme: {0}'.format(use_metapath_scheme),
                                   metapath_scheme='The specified metapath scheme: {0}'.format(metapath_scheme),
                                   constraint_type='Use node type: {0}'.format(constraint_type),
                                   just_type='Use JUST algorithm: {0}'.format(just_type),
                                   burn_in_phase='Burn in phase count: {0}'.format(self.burn_in_phase),
                                   burn_in_input_size = 'Subsampling size of the number of walks and length for burn in phase: {0}'.format(self.burn_in_input_size))
            time.sleep(2)
            
        init_node_prob, type2index, type2prob = self.__init_probability(hin)
        hin.type2index = type2index
        hin.type2prob = type2prob
        hin.trans_metapath_scheme = use_metapath_scheme
        hin.trans_constraint_type = constraint_type
        hin.trans_just_type = just_type
        hin.q = self.q
        hin.p = self.p
        hin.learning_rate = self.learning_rate
        hin.num_walks = self.num_walks
        hin.walk_length = self.walk_length

        print('\t>> Calculate initial transition probabilities...')
        logger.info('\t>> Calculate initial transition probabilities...')
        N = (hin.number_of_nodes(), hin.number_of_nodes())
        trans_prob = lil_matrix((N[0], N[0]))
        trans_prob.data[:] = EPSILON
        for curr_node, curr_node_data in hin.nodes(data=True):
            neigh_curr_node = np.array(
                [hin.nodes[edge[1]]['mapped_idx'] for edge in hin.edges(curr_node)])
            trans_prob[curr_node_data['mapped_idx'], neigh_curr_node] = 1
        trans_prob = lil_matrix(trans_prob.multiply(1 / trans_prob.sum(1)))

        print('\t>> Calculate transition probabilities...')
        logger.info('\t>> Calculate transition probabilities...')
        for burn_in_count in np.arange(start=1, stop=burn_in_phase + 1):
            if burn_in_count > 1:
                desc = '\t\t## Burn in phase {0} (out of {1})...{2}'.format(burn_in_count, burn_in_phase, 20 * ' ')
                print(desc)
                logger.info(desc)
            for node_idx, node_data in enumerate(hin.nodes(data=True)):
                trans_prob = self._walks_per_node(node_idx=node_idx, node_curr=node_data[0],
                                                   node_curr_data=node_data[1], hin=hin, just_memory_size=just_memory_size, 
                                                   trans_prob=trans_prob, 
                                                   burn_in_phase=True)
        node_prob = trans_prob.T.dot(init_node_prob)
        results = node_prob.sum()
        node_prob = node_prob.multiply(1 / results)
        hin.trans_prob = trans_prob
        for node in hin.nodes(data=True):
            attrs = {node[0]: {'weight': node_prob[node[1]['mapped_idx']]}}
            nx.set_node_attributes(hin, attrs)
        save_data(data=hin, file_name=save_file_name + '.pkl', save_path=ospath,
                  tag='heterogeneous information network', mode='wb')

        print('\t>> Generate walks...')
        logger.info('\t>> Generate walks...')
        if os.path.exists(os.path.join(dspath, 'X_' + save_file_name + '.txt')):
            os.remove(path=os.path.join(dspath, 'X_' + save_file_name + '.txt'))
        pool = Pool(processes=self.num_jobs)
        results = [pool.apply_async(self._walks_per_node, args=(node_idx, node_data[0], node_data[1], 
                                                                hin, just_memory_size, trans_prob, dspath, 'X_' + save_file_name + '.txt', False))
                    for node_idx, node_data in enumerate(hin.nodes(data=True))]
        output = [p.get() for p in results]
