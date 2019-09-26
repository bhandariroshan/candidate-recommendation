Note
---------------------
edges.txt is the real world graph downloaded from snap https://snap.stanford.edu/data/amazon0302.html . 
This file corresponds to amazon's copurchase data based on 'people who bought this also bought this' feature.


Instructions
---------------------
1. Extract the gzip file 
2. install networkx, scipy, numpy and matplotlib
3. Open Terminal and Python console in that directory
4. Loading data to network (nodes.txt and edges.txt)
>> from main import NetworkLoader
>> network_loader = NetworkLoader(edge_file_name='edges.txt', node_file_name='nodes.txt')

5. To plot subgraph of the network loaded, type following
>> network_loader.plot_a_subgrapsh()

-- This will save you a graph.png file in the folder

6. To compute eigen values and find maximum and minimum eigen values
>> network_loader.compute_eigen_values()

7. To find centralities of the network
>> network_loader.find_network_centralities()

8. To save plot of degrees of the graph or network
>> network_loader.save_degrees_graph_plot()

-- This will save you a degree_distribution.png, in_degree_distribution.png, out_degree_distribution.png file in the folder containing code

9. To find and plot average, min and max degrees of the network graph
>> network_loader.find_and_plot_average_min_and_max_degree_of_graph()

-- This will save you a  degree.png file in the folder containing code

10. To crawl the network and generate recommendation for products
>> network_loader.crawl_network_to_generate_recommendation()

-- This will crawl the network using different algorith (BFS, DFS, RFS) and generate a product suggestion file (traversal.csv) that contaings id of the product and the time required to make the suggestion when we give different products (hard coded in the case)

-- This will also generate time.png file that contains bar graph containing average time of computation for making product recommendation.
