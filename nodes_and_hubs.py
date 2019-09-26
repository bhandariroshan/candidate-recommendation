import sys
import scipy
import math
from scipy.io import mmread
import networkx as nx
from networkx.convert_matrix import from_scipy_sparse_matrix
import multiprocessing
from multiprocessing import Pool


class NetworkManipulator(object):
    def __init__(self, 
        edge_file_name=None, 
        steps=3,
        normalize=True
    ):
        self.steps = steps
        self.graph = nx.DiGraph()
        self.normalize = bool(normalize)
        self.matrix_file_name = matrix_file_name
        self.edge_file_name = edge_file_name

        if self.edge_file_name:
            self.load_network_from_edgelist_file(self.edge_file_name)

        else:
            raise ('Nothing to do. Please check.')          

    def load_network_from_edgelist_file(self, edge_file_name):
        if '.mtx' in edge_file_name or '.mm' in edge_file_name:
            self.load_network_from_scipy_matrix()
        else:
            fh=open(edge_file_name, 'rb')
            self.graph = nx.read_edgelist(fh, create_using=nx.DiGraph())

    def load_network_from_scipy_matrix(self):
        scipy_matrix = mmread(self.matrix_file_name)
        self.graph = from_scipy_sparse_matrix(scipy_matrix, create_using=nx.DiGraph())

    def print_network_with_hubs_and_authorities(self):
        network = []
        for each_node, data in self.graph.nodes(data=True):
            data['node'] = each_node
            network.append(data)

        auth_network = sorted(network,  key=lambda k: k['auth'], reverse=True)

        for each_node in auth_network:
            print(each_node['node'], each_node['auth'], each_node['hub'])

    def do_parallel_in_nodes_processing(self, node, data, norm):
        data['auth'] = 0
        neighbors = self.graph.in_edges(node, data=True)
        for in_node, self_node, in_edge_data in neighbors:
            data['auth'] += self.graph.nodes[in_node]['hub']

        if self.normalize:
            norm += math.pow(data['auth'], 2) # calculate the sum of the squared auth values to normalise
        
        return norm

    def do_parallel_in_hub_processing(self, node, data, norm):
        data['hub'] = 0
        neighbors = self.graph.in_edges(node, data=True)
        for in_node, self_node, in_edge_data in neighbors:
            data['hub'] += self.graph.nodes[in_node]['auth']

        if self.normalize:
            norm += math.pow(data['hub'], 2) # calculate the sum of the squared auth values to normalise
        
        return norm

    def find_hubs_and_authorities(self):
        cpu_count = multiprocessing.cpu_count()

        for node, data in self.graph.nodes(data=True):
            data['auth'] = 1
            data['hub'] = 1

        for i in range(int(self.steps)):
            global norm 
            norm = 0

            def add_norm(add_val):
                global norm
                norm += add_val

            pool = Pool(cpu_count)
            results = [
            pool.apply_async(
                self.do_parallel_in_nodes_processing, (node, data, norm), callback = add_norm
                ) for node, data in self.graph.nodes(data=True)
            ]
            
            [res.get() for res in results]

            if self.normalize:
                norm = math.sqrt(norm)
                for node, data in self.graph.nodes(data=True):
                    data['auth'] = round(data['auth'] / norm, 3)
            
            norm = 0
            pool = Pool(cpu_count)
            results = [
            pool.apply_async(
                self.do_parallel_in_hub_processing, (node, data, norm), callback = add_norm
                ) for node, data in self.graph.nodes(data=True)
            ]
            
            [res.get() for res in results]

            if self.normalize:
                norm = math.sqrt(norm)
                for node, data in self.graph.nodes(data=True):
                    data['hub'] = round(data['hub'] / norm, 3)
        
        self.print_network_with_hubs_and_authorities()


if __name__ == "__main__":  
    # execute only if run as a script
    matrix_file_name = sys.argv[1]
    steps = sys.argv[2]
    normalize = int(sys.argv[3])

    nm = NetworkManipulator(
        # matrix_file_name='delaunay_n14.mtx', 
        edge_file_name=matrix_file_name,
        steps=steps,
        normalize=normalize
    )
    nm.find_hubs_and_authorities()