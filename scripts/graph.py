from collections import Counter, defaultdict, deque
from tqdm import tqdm

import fibheap as fib
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    import functionality as fn
except:
    import scripts.functionality as fn



class Vertex:
    """
    Class to handle the vertices of a graph. 
    Product of a previous implementation, probably garbage code.
    """

    def __init__(self, name):
        self.name = name
        self.dist = float('inf')
        self.pred = None


    @classmethod
    def from_list(cls, list_):
        return {v: cls(v) for v in list_}


    def __lt__(self, other):
        return self.dist < other.dist

    
    def __le__(self, other):
        return self.dist <= other.dist


    def __eq__(self, other):
        return self.dist == other.dist


    def __add__(self, other):
        return self.dist + 1


    def __hash__(self):
        return hash(self.name)


    def __repr__(self):
        return self.__str__()


    def __str__(self):
        return f'{self.name}: dist={self.dist}, pred={self.pred}'


class Graph:

    def __init__(self):
        """
        Initialize the Graph object

        :attribute edges : dictionary of vertices (keys) to sets of vertices (values)
        :attribute induced_subgraph:
        """
        self._adj_list = defaultdict(set)


    @classmethod
    def from_dict(cls, dict_):
        """
        Create a new Graph object from a dictionary

        :param dict_ : dictionary of vertices (keys) to sets of vertices (values)
        :return : new Graph instance
        """
        g = cls()
        g._adj_list.update({k:set(v) for k, v in dict_.items()})
        vertices = list(g._adj_list.keys())
        for v in vertices:
            for u in g._adj_list[v]:
                g.add_vertex(u)     # make sure that all the vertices of the graph are in the adjacency list:
                                    # without this step, a sink node wouldn't appear in g._adj_list
        return g


    @property
    def n_vertices_(self):
        """
        Read-only property, number of vertices in the graph
        """
        return len(self._adj_list)


    @property
    def n_edges_(self):
        """
        Read-only property, number of edges in the graph
        """
        n = 0
        for v in self._adj_list:
            n += len(self._adj_list[v]) 
        return n


    @property
    def density_(self):
        """
        Read-only property, density of the graph: |E| / [ |V| * (|V| - 1) ]
        """
        n_v = self.n_vertices_
        return np.format_float_scientific(self.n_edges_ / (n_v * (n_v - 1)), precision=3)


    def get_vertices(self):
        """
        Get the vertices of the graph

        :return : list of vertices
        """
        return list(self._adj_list.keys())


    def get_edges(self):
        """
        Get the edges of the graph

        :return : list of (source, destination) tuples
        """
        return [(v, u) for v in self._adj_list.keys() for u in self._adj_list[v]]


    def add_edge(self, v, u):
        """
        Add an edge to the graph

        :param v : source vertex
        :param u : destination vertex

        :return : 
        """
        self._adj_list[v].add(u)


    def add_vertex(self, v):
        """
        Add a vertex to the graph

        :param v : new vertex
        
        :return : 
        """
        self._adj_list[v]


    def in_degree(self, v=None):
        """
        Compute the in-degree of a node v

        :param v : a node of the graph
        :return in_d : int
        """
        if v is None:
            degrees = defaultdict.fromkeys(self._adj_list.keys(), 0)
            for v in degrees:
                for u in self._adj_list[v]:
                    degrees[u] += 1
            return dict(degrees)
        else:
            in_d = 0
            for u in self._adj_list:
                if v in self._adj_list[u]:
                    in_d += 1
            return in_d


    def out_degree(self, v=None):
        """
        Compute the out-degree of a node v

        :param v : a node of the graph
        :return  : int
        """
        if v is None:
            degrees = dict.fromkeys(self._adj_list.keys(), 0)
            for v in degrees:
                degrees[v] = len(self._adj_list[v])
            return degrees
        return len(self._adj_list[v])


    def degree(self, v=None):
        """
        Compute the degree of a node v

        :param v : a node of the graph
        :return  : int
        """
        if v is None:
            in_deg = self.in_degree()
            out_deg = self.out_degree()
            degrees = dict.fromkeys(self._adj_list.keys(), 0)
            for v in degrees:
                degrees[v] = in_deg[v] + out_deg[v]
            return degrees
        return self.in_degree(v) + self.out_degree(v)


    def degree_distro(self, normalize=False):
        """
        Return the degree distribution of the graph

        :param normalize : normalize the degree distribution
        :return : degree d to number of vertices with degree d, defaultdict
        """
        degree_dist = Counter(self.degree().values())
        if normalize:
            n_v = self.n_vertices_
            degree_dist = {d:round(degree_dist[d] / n_v, 8) for d in degree_dist}
        return dict(degree_dist)

    
    # TODO: make the function more flexible, i.e. allow to control and change more parameters, provide more visualizations options
    def plot_degree_distro(self, normalize=True, log=True, interval=None):
        """
        Plot the degree distribution of the graph

        :param normalize : normalize the degree distribution dividing by the number of vertices
        :param log : if True, plot the distribution in a log-log plot
        :param interval : interval [a,b) of degrees to be considered
        :return 
        """
        degree_dist = self.degree_distro(normalize)
        if isinstance(interval, tuple):
            degree_dist = {degree:value for degree,value in degree_dist.items()\
                                        if degree>=interval[0] and degree<interval[1]}
        plt.figure(figsize=(16,10))
        plt.bar(list(degree_dist.keys()), degree_dist.values())
        if log:
            plt.yscale('log')
            plt.xscale('log')
            plt.title('Log-log degree distribution')
        else:
            plt.title('Degree distribution')
        plt.xlabel('Degree')
        plt.ylabel('Number of nodes')
        plt.grid()
        plt.show()


    # TODO: make the function more flexible; extend it to other types of graphs
    def _dijkstra(self, src, pred = None):
        """
        Internal routine to compute the shortest paths from a source according to the Dijkstra algorithm.
        Implementation only for unweighted graphs

        :param src : source vertex
        :param pred : dictionary of vertex:predecessor

        :return : dictionary of distances between the source vertex and all the other vertices
        """
        dist = {}
        processed = defaultdict(lambda: float('inf'))
        queue = []
        heapq.heappush(queue, (0, src))
        with tqdm() as pbar:
            while queue:
                pbar.update(1)
                (d, u) = heapq.heappop(queue)
                if u in dist:
                    continue
                dist[u] = d
                for v in self._adj_list[u]:
                    if v not in dist:
                        if processed[v] > dist[u] + 1:
                            processed[v] = dist[u] + 1
                            if pred is not None:
                                pred[v] = u
                            heapq.heappush(queue, (processed[v], v))
        return dist
                
    
    def shortest_path(self, src, rec_path=False):
        """
        Compute the shortest paths between the source vertex and all the vertices in the graph.

        :param src : source vertex
        :param rec_path (optional) : save the predecessor of each node

        :return : dictionary of distances between the source vertex and all the other vertices
        :return : if rec_path==True, return also a dictionary of vertices predecessors
        """
        if rec_path:
            pred = defaultdict(lambda: None)
            return self._dijkstra(src, pred=pred), pred
        return self._dijkstra(src)
        

    def all_pairs_shortest_path(self, vertices=None):
        """
        Compute the shortest paths between a set of vertices and all the other vertices in the graph.

        :param vertices : subset of vertices in the graph from which compute the shortest path;
                        if vertices==None, compute the the shortest paths between each pair of vertices in the graph.

        :return : dictionary of distances; src_vertex: {all_vertices: dist}
        """
        distances = dict()
        if vertices is None:
            for v in self._adj_list:
                distances[v] = self._dijkstra(v)
        else:
            for v in vertices:
                distances[v] = self._dijkstra(v)
        return distances


    def category_distance(self, category, categories):
        """
        Compute the distances between a given category and all the others.

        :param category : category (name) from which compute the distances
        :param categories : dictionary of the categories in the graph

        :return : list of categories' names sorted by their distances from the source category
        """
        cat_vert = categories.pop(category, None)
        v_dist = self.all_pairs_shortest_path(cat_vert)
        cat_dist = np.zeros(len(categories))
        cat_names = []
        for idx, c in enumerate(categories.keys()):
            cat_dist[idx] = np.median(np.array([ v_dist[u][v] if v in v_dist[u] else float('inf') for u in cat_vert for v in categories[c] ]))
            cat_names.append(c)
        return list(np.array(cat_names)[np.argsort(cat_dist)])
            

    def __repr__(self):
        """
        Represent the graph as the list of its edges
        """
        return str(self.get_edges())


    @staticmethod
    def plot_graph(graph, with_labels=True, node_size=100):
        """
        Method to visualize graph or sub_graph

        :param graph: graph to be visualized
        :param with_labels: bool to add labels or not
        :param node_size: node size
        :return: Plot
        """
        g = nx.DiGraph(graph)
        plt.figure(figsize=(12, 8))
        plt.clf()
        nx.draw(g, with_labels=with_labels, node_size=node_size)
        plt.show()


    def pages_in_click(self, initial_page, num_clicks, print_=False):
        """
        Given a graph, an initial starting point and the number of clicks, how many, and which pages will we be able to
        visit?

        :param initial_page: Page we will be starting out from
        :param num_clicks: Number of clicks we are willing to do
        :param print_: Bool to visualize some of the outputs or not
        :return: Pages seen  with the given number of clicks
        """

        # This will be a list of all the pages that we are able to visit during our clicks
        pages_visited = set()
        # This will be a list of articles that we are able to reach at the ith click. We will use this list to check the
        # articles that we can reach in the i+1th click
        queue = set([initial_page])

        # Placeholder to keep track of the clicks we are doing
        clicks = 0
        # Interrupt the loop once we reach the required number of clicks
        while clicks < num_clicks:
            # List of elements that will be used for the next loop
            new_queue = set()
            # List of elements that don't have any out-node
            last_nodes = set()
            # Loop over all the pages of the current click
            for node in queue:
                if print_:
                    print(node)
                # If a given node has target node, include the out-nodes into the new_queue list
                if bool(self._adj_list[node]):
                    new_queue.update(self._adj_list[node])
                # If a given node has no target node, include it in the last_nodes list (this list will not be used to)
                # for further inspection but we will have to consider it as an article that has been seen
                else:
                    last_nodes.update([node])

            if print_:
                print(new_queue)

            # Update queue as the new pages to explore
            queue = new_queue
            # Update pages_seen with the pages that have been seen in this click (pages that still have out-nodes and
            # pages that end at that node)
            pages_visited.update(new_queue | last_nodes)
            # Update the number of clicks done
            clicks += 1

        # Return the unique pages
        return set(pages_visited)


    # TODO:  Use class methods to  construct new induced-subgraph. Issue is that the methods construct over the 
    #  self.edge and we don't  want that
    # TODO: See how to return the new class
    def generate_induced_subgraph(self, vertices):
        """
        Given a set of vertices, compute it's induced subgraph

        :param vertices: Set of vertices
        :return: Store induced sub_graph in class
        """
        induced_subgraph = defaultdict(set)
        for vertex in vertices:
            induced_subgraph[vertex] = vertices.intersection(self._adj_list[vertex])

        return Graph(induced_subgraph)


    # TODO: probably not needed method
    def bfs(self, start):
        """
        Function to compute the bfs at any starting point

        :param start: Initial page
        :return: Pages that can be visited from that starting point
        """
        visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.update([node])
                neighbours = self._adj_list[node]
                for neighbour in neighbours:
                    queue.append(neighbour)
        return visited


    def min_cut(self, article1, article2):
        """
        Given two random articles, get the minimum number of edges that need to be breaken down to dut the link
        between both articles.

        This algorithm will follow the logic followed by the Edmonds-Karp Algorithm (max flow is equal to min cut),
        which is an improvement on the Ford-Fulkerson Algorithm

        :param article1: Integer representing an article (source article)
        :param article2: Integer representing an article (sink article)
        :return: Integer (number of edges)
        """
        x = self.induced_subgraph
        min_cut = 0

        return min_cut