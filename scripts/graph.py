import copy
import heapq
import random
from collections import Counter, defaultdict, deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as plty
from tqdm import tqdm

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
        self._capacity = defaultdict(dict)

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


    def get_edges(self, output='tuple'):
        """
        Get the edges of the graph
        :param output: Output can be a list of tuples ('tuple'), a list of lists ('list') or a dictionary with lists
            ('dict')
        :return : list of (source, destination) tuples
        """
        if output == 'tuple':
            return [(v, u) for v in self._adj_list.keys() for u in self._adj_list[v]]
        else:
            edge_list = [[v, u] for v in self._adj_list.keys() for u in self._adj_list[v]]
            if output == 'list':
                return edge_list
            else:
                return {i: edge for i, edge in enumerate(edge_list)}

    def add_edge(self, v, u):
        """
        Add an edge to the graph

        :param v : source vertex
        :param u : destination vertex

        :return : 
        """
        self._adj_list[v].add(u)
        self.add_vertex(u)


    def add_vertex(self, v):
        """
        Add a vertex to the graph

        :param v : new vertex
        
        :return : 
        """
        self._adj_list[v]

    @staticmethod
    def from_edges_to_graph(edges):
        """
        Given a list of tuples with vertices, generate a graph as a dict

        :param edges: List of tuples with vertices
        :return: Dict graph
        """
        dict_stupid = defaultdict(set)
        for item in edges:
            dict_stupid[item[0]].update([item[1]])

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


    # TODO: make the function more flexible, i.e. allow to control and change more parameters, provide more
    #  visualizations options
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


    def _bfs(self, src, targets_v=None, pred=None):
        """
        Function to compute the bfs at any starting point

        :param src : Initial page
        :param targets_v : vertices of interest; the function stops when all the vertices in targets_v have been explored
        :param pred : dictionary of vertex:predecessor

        :return : dictionary of distances between the source vertex and all the other vertices
        """
        if targets_v is None:
            dest_v = set([None])
        else:
            dest_v = set(targets_v)
        queue = deque([src])
        dist = {src:0}

        while queue and dest_v:
            u = queue.popleft()
            for v in self._adj_list[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                    dest_v.discard(v)
                    if pred is not None:
                        pred[v] = u
        return dist


    def _dijkstra(self, src, targets_v=None, pred=None):
        """
        Internal routine to compute the shortest paths from a source according to the Dijkstra algorithm.
        Implementation only for unweighted graphs

        :param src : source vertex
        :param targets_v : vertices of interest; the function stops when all the vertices in targets_v have been explored
        :param pred : dictionary of vertex:predecessor

        :return : dictionary of distances between the source vertex and all the other vertices
        """
        if targets_v is None:
            dest_v = set([None])
        else:
            dest_v = set(targets_v)
        dist = {}
        processed = defaultdict(lambda: float('inf'))
        queue = []
        heapq.heappush(queue, (0, src))
        # with tqdm() as pbar:
        while queue and dest_v:
            # pbar.update(1)
            (d, u) = heapq.heappop(queue)
            if u in dist:
                continue
            dist[u] = d
            for v in self._adj_list[u]:
                if v not in dist:
                    if processed[v] > dist[u] + 1:
                        processed[v] = dist[u] + 1
                        heapq.heappush(queue, (processed[v], v))
                        dest_v.discard(v)
                        if pred is not None:
                            pred[v] = u
        return dist


    def shortest_path(self, src, targets_v=None, rec_path=False, how='bfs'):
        """
        Compute the shortest paths between the source vertex and all the vertices in the graph.

        :param src : source vertex
        :param targets_v (optional) : vertices of interest; the function stops when all the vertices in targets_v have been explored
        :param rec_path (optional) : save the predecessor of each node
        :param how : algorithm for computing the shortest paths; 'bfs' or 'dijkstra'

        :return : dictionary of distances between the source vertex and all the other vertices
        :return : if rec_path==True, return also a dictionary of vertices predecessors
        """
        algorithm = {'bfs':self._bfs, 'dijkstra':self._dijkstra}[how]
        if rec_path:
            pred = defaultdict(lambda: None)
            return algorithm(src, targets_v=targets_v, pred=pred), pred
        return algorithm(src, targets_v=targets_v)


    def all_pairs_shortest_path(self, vertices=None, only_targets=False, how='bfs'):
        """
        Compute the shortest paths between a set of vertices and all the other vertices in the graph.

        :param vertices : subset of vertices in the graph from which compute the shortest path;
                        if vertices==None, compute the the shortest paths between each pair of vertices in the graph.
        :param how : algorithm for computing the shortest paths; 'bfs' or 'dijkstra'

        :return : dictionary of distances; src_vertex: {all_vertices: dist}
        """
        distances = dict()
        if vertices is None:
            for v in tqdm(self._adj_list):
                distances[v] = self.shortest_path(v, how=how)
        else:
            if only_targets:
                for v in tqdm(vertices):
                    distances[v] = self.shortest_path(v, targets_v=vertices, how=how)
            else:
                for v in tqdm(vertices):
                    distances[v] = self.shortest_path(v, how=how)
        return distances


    def category_distance(self, category, categories):
        """
        Compute the distances between a given category and all the others.

        :param category : category (name) from which compute the distances
        :param categories : dictionary of the categories in the graph

        :return : list of categories' names sorted by their distances from the source category
        """
        categories_copy = categories.copy()
        cat_vert = categories_copy.pop(category, None)
        v_dist = self.all_pairs_shortest_path(cat_vert)
        cat_dist = np.zeros(len(categories_copy))
        cat_names = []
        for idx, c in enumerate(categories_copy.keys()):
            cat_dist[idx] = np.median(np.array([ v_dist[u][v] if v in v_dist[u] else float('inf') for u in cat_vert for v in categories_copy[c] ]))
            cat_names.append((c, cat_dist[idx]))
        return list(np.array(cat_names)[np.argsort(cat_dist)])


    def dist_weighted_graph(self, vertices=None, distances=None):
        """
        Compute a new weighted graph having the vertices in vertices (in the graph if vertices is None)
        connected by edges weighted with the minimum distsnce between each pair of vertices

        :param vertices (optional) : subset of vertices in the graph
        :param distances (optional) : dictionary of precomputed distances

        :return : new WeightedGraph instance
        """
        if distances is None:
            distances = self.all_pairs_shortest_path(vertices, only_targets=True, how='bfs')
        wg = WeightedGraph()
        for u in distances.keys():
            wg.add_vertex(u)
            for v in distances[u].keys():
                if v in distances.keys() and distances[u][v] != 0:
                    wg.add_edge(u, v, distances[u][v])
        return wg


    def minimum_cat_walk(self, cat_vertices):
        """
        Compute an approximation (if possible) of the shortest walk across all the vertices in cat_vertices,
        starting from the most central vertex in cat_vertices. Closeness is assumed as the metric of centrality.

        :param cat_vertices : vertices in the target category

        :return : 
        """
        print('Calculating the distances between each pair of vertices in the graph...')
        distances = self.all_pairs_shortest_path(cat_vertices, only_targets=True)
        print('Calculating the source vertex according to the closeness centrality measure...')
        src = -1
        min_dist = float('inf')
        for v in distances.keys():
            dist_v = 0
            for u in distances[v].keys():
                if u in distances.keys():
                    dist_v += distances[v][u]
            if dist_v < min_dist:
                src = v
                min_dist = dist_v
        wg = self.dist_weighted_graph(distances=distances)
        cost = wg.nearest_neighbor(src)
        print(f'The source vertex is:\t{src}')
        if cost < float('inf'):
            print(f'The cost of the approximated minimum walk is:\t{cost}')
        else:
            print('Not possible!')

    def __repr__(self):
        """
        Represent the graph as the list of its edges
        """
        return str(self.get_edges())

    def plot_graph(self, with_labels=True, node_size=100):
        """
        Method to visualize graph or sub_graph

        :param graph: graph to be visualized
        :param with_labels: bool to add labels or not
        :param node_size: node size
        :return: Plot
        """
        g = nx.DiGraph(self._adj_list)
        plt.figure(figsize=(12, 8))
        plt.clf()
        nx.draw(g, with_labels=with_labels, node_size=node_size)
        plt.show()


    # Question 2
    def pages_in_click(self, initial_page, num_clicks):
        """
        Given an initial starting point and the number of clicks, how many pages, and which ones will we be able to
        visit?

        :param initial_page: Page we will be starting out from
        :param num_clicks: Number of clicks we are willing to do
        :return: Pages seen  with the given number of clicks
        """

        # This will be a list of all the pages that we are able to visit during our clicks
        pages_visited = set()
        # This will be a list of articles that we are able to reach at the ith click. We will use this list to check the
        # articles that we can reach in the i+1th click
        queue = set([initial_page])

        # Placeholder to keep track of the clicks we have done
        clicks = 0
        # Interrupt the loop once we reach the required number of clicks
        while clicks < num_clicks:
            # List of elements that will be used for the next loop
            new_queue = set()
            # List of elements that don't have any out-node
            last_nodes = set()
            # Loop over all the pages of the current click
            for node in queue:
                # If a given node has target node, include the out-nodes into the new_queue list
                if bool(self._adj_list[node]):
                    new_queue.update(self._adj_list[node])
                # If a given node has no target node, include it in the last_nodes list (this list will not be used to)
                # for further inspection but we will have to consider it as an article that has been seen
                else:
                    last_nodes.update([node])

            # Update queue as the new pages to explore
            queue = new_queue
            # Update pages_seen with the pages that have been seen in this click (pages that still have out-nodes and
            # pages that end at that node)
            pages_visited.update(new_queue | last_nodes)
            # Update the number of clicks done
            clicks += 1

        # Return the unique pages
        return list(set(pages_visited))

    # Question 4
    def generate_induced_subgraph(self, vertices):
        """
        Given a set of vertices, compute it's induced subgraph

        :param vertices: Set of vertices
        :return: Store induced sub_graph in class
        """
        induced_subgraph = defaultdict(set)
        for vertex in vertices:
            induced_subgraph[vertex] = vertices.intersection(self._adj_list[vertex])
            induced_subgraph = {k: v for k, v in induced_subgraph.items() if v}

        return Graph.from_dict(induced_subgraph)

    def build_capacity(self):
        """
        The max-flow method requires a residual graph to work. This can also be seen as an attribute of the class,
        therefore we will include it as a self.

        :return: Generate attribute
        """
        for k, v in self._adj_list.items():
            self._capacity[k].update({k: 1 for k in v})

    def compute_augmented_path(self, capacity, source, sink):
        """
        Function to compute the augmented paths required for the max flow algo (which is equal to the min cut value)
        The idea behind computing this augmented  path is looking for a path that has not been visited before and with
        a capacity higher than 0
        """
        visited = set()
        queue = deque([source])
        parent_map = []
        # If the augmented path cannot continue it can be because the node has already been visited or because it's
        # capacity is smaller than zero. If the second is the case this might hint that this edge will need to be cut
        # in order to disconnect two nodes
        funnel_node = []

        count = 0
        while queue:
            count += 1
            node = queue.popleft()
            if node not in visited:
                visited.update([node])
                neighbours = self._adj_list[node]
                for neighbour in neighbours:
                    if (neighbour not in visited) & (capacity[node][neighbour] > 0):
                        queue.append(neighbour)
                        parent_map.append([neighbour, node])
                        if neighbour == sink:
                            return parent_map, funnel_node
                    else:
                        if (capacity[node][neighbour] == 0) & (node != source):
                            funnel_node.append([node, neighbour])
                        continue
        return None, funnel_node

    @staticmethod
    def construct_flow(parent_map, sink):
        """
        Given a parent map (output of the compute augmented path function) construct a flow that goes from the sink to
        the initial point
        """
        flow = [sink]
        for row in parent_map[::-1]:
            if row[0] == flow[-1]:
                flow.append(row[1])
        return flow[::-1]

    @staticmethod
    def compute_flow_adjust_capacity(flow, capacity):
        """
        Given a flow and the capacity values for each, compute the flow value (min over the given augmented path), and
        re-adjust the capacity dictionary for the next iteration

        """

        flow_value = min([capacity[flow[i]][flow[i + 1]] for i in range(0, len(flow) - 1)])
        for i in range(len(flow) - 1):
            capacity[flow[i]][flow[i + 1]] = capacity[flow[i]][flow[i + 1]] - flow_value

        return flow_value, capacity

    def max_flow_func(self, source, sink):
        """
        Function to compute the max flow over a given graph, capacity, initial source value and sink value

        """
        self.build_capacity()

        max_flow = 0
        capacity = copy.deepcopy(self._capacity)
        flow_storage = []

        while True:
            print('===================')
            augmented_path, funnel_node = self.compute_augmented_path(capacity=capacity,
                                                                      source=source, sink=sink)

            if not augmented_path:
                funnel_edge = [(str(edge[0]) + ' --> ' + str(edge[1])) for edge in funnel_node][:max_flow]
                test = []
                for edge_funnel in funnel_node:
                    for edge_storage in flow_storage:
                        if any([edge_storage[i:i + 2] == edge_funnel for i in range(len(edge_storage) - 1)]):
                            test.append(edge_storage)
                final = [x for x in flow_storage if x not in test]
                other_edge = [(str(flow_path[-2]) + ' --> ' + str(flow_path[-1])) for flow_path in final]
                edges_to_cut = funnel_edge + other_edge
                return edges_to_cut, max_flow

            flow = self.construct_flow(augmented_path, sink)
            print('Flow:', flow)
            flow_storage.append(flow)
            flow_value, capacity = self.compute_flow_adjust_capacity(flow, capacity)
            max_flow += flow_value

    # Question 6
    def category_model_network_weighted(self, article_category_dict):
        """
        Given a dictionary that maps article integer to category, return a graph class with the new category model
        network. This network will mainly be used to later compute the page rank score for each category and will also
        be weighted (weight are the number of links that go from one category to the next)

        :param article_category_dict: Dictionary that maps articles to categories
        :return: New graph class with the nodes this time the categories, and the edges are links between categories
            (one link between categories implies that at least one article of that category has a link to the other
            category)
        """
        category_adj_list = defaultdict(dict)
        for k, v in self._adj_list.items():
            for node in v:
                try:
                    # noinspection PyTypeChecker
                    category_adj_list[article_category_dict[k]][article_category_dict[node]] = \
                        category_adj_list[article_category_dict[k]][article_category_dict[node]] + 1
                except KeyError:
                    category_adj_list[article_category_dict[k]].update({article_category_dict[node]: 1})

        return category_adj_list
        # TODO: Implement such that we can also have a weighted graph
        # return Graph.from_dict_weighted(category_adj_list)

    def page_rank(self, weighted_graph, damping_factor=.85, max_iter=100, tolerance=0.01, top=None):
        """
        Method that computes the page rank of a given weigthed and directed graph

        :param weighted_graph: Weigthed graph
        :param damping_factor: Probability of stopping to surf at any given moment (popular to include now a days to
            model the probability that any user immediately stops using internet)
        :param max_iter: Maximum iterations to find a good convergence of the page rank score
        :param tolerance: Tolerance level required for the iteration to stop earlier
        :param top:  Return top categories (int) based on page rank or return all (None)
        :return: Page Rank score for all nodes

        Example:

        dummy_graph  = {'B': {'C': 1},
                        'C': {'B': 6},
                        'D': {'A': 1, 'B': 4},
                        'E': {'B': 5, 'D': 2, 'F': 3},
                        'F': {'E': 4}})

        outlink_dict = {'A': {'D': 1/5},
                        'B': {'C': 6/6, 'D': 4/5, 'E': 5/10},
                        'C': {'B': 1/1},
                        'D': {'E': 2/10},
                        'E': {'F': 4/4},
                        'F': {'E': 3/10}}
        """

        # Step 1: Generate dictionary with outlink probabilities [probability of  going to a page (key of this dict)
        # from the pages in the values]
        outlink_dict = defaultdict(dict)
        for k, v in weighted_graph.items():
            for i in v:
                try:
                    outlink_dict[i][k] = outlink_dict[i][k] + weighted_graph[k][i] / sum(weighted_graph[k].values())
                except:
                    outlink_dict[i].update({k: weighted_graph[k][i] / sum(weighted_graph[k].values())})

        # Number of nodes
        N = len(outlink_dict)

        # Initial Page rank (same probability of starting out at any random node)
        page_rank_score_t = {k: 1/N for k, v in outlink_dict.items()}
        # Updated page rank score. This is what we will update constantly.Once these two are similar, convergance will
        # be reached
        page_rank_score_t_1 = {}
        i = 0
        while i < max_iter:
            i += 1
            for key, value in outlink_dict.items():
                temp = sum(
                    {k: v * page_rank_score_t[k] for k, v in outlink_dict[key].items() if k in page_rank_score_t}.values())
                page_rank_score_t_1[key] = ((1 - damping_factor) / N) + damping_factor * temp
            factor = 1 / sum(page_rank_score_t_1.values())

            # Normalize updated page rank score accordingly
            for k in page_rank_score_t_1:
                page_rank_score_t_1[k] = page_rank_score_t_1[k] * factor

            diff = np.sqrt(
                sum((np.array(list(page_rank_score_t.values())) - np.array(list(page_rank_score_t_1.values()))) ** 2))

            if diff < tolerance:
                print('Converged in', i, 'iterations')
                break

            if i != max_iter:
                page_rank_score_t = page_rank_score_t_1
                page_rank_score_t_1 = {}
        final_dict = dict(sorted(page_rank_score_t_1.items(), key=lambda item: item[1], reverse=True))
        if not top:
            return final_dict
        else:
            top_dict = {}
            for i, k in enumerate(final_dict):
                top_dict[k] = final_dict[k]
                if i == top:
                    break
            return pd.DataFrame.from_dict(top_dict, orient='index', columns=['Page Rank Score'])

    # ===== Unused Methods =======
    @staticmethod
    def get_random_edge(edges, source, sink):
        """
        Function that randomly returns two connected vertices

        :param edges: List of list containing the edges over which to pick
        :param source & sink: These are the two vertices which we want to disconnect, therefore they cannot be v2
        (v2 is the node that will disappear so to say)
        :return: two vertices
        """

        v1, v2 = random.choice(list(edges.values()))
        while (v2 == source) | (v2 == sink):
            v1, v2 = random.choice(list(edges.values()))

        return v1, v2

    @staticmethod
    def contract_node(edges, v1, v2):
        """
        Function to contract two nodes into one. The logic is that we will delete the edge between two nodes, and
        basically merge the node v2 into v1 (v1 will inherit all v2's edges).
        """

        # Make v1 inherit all of v2's edges
        contracted = {}
        for k, v in edges.items():
            if v2 in v:
                contracted[k] = [v1 if v2 == ele else ele for ele in v]
            else:
                contracted[k] = v

        # Remove edges with same in and out node
        contracted = {k: v for k, v in contracted.items() if v[0] != v[1]}

        return contracted

    def category_model_network(self, article_category_dict):
        """
        Given a dictionary that maps article integer to category, return a graph class with the new category model
        network. This network will mainly be used to later compute the page rank score for each category

        :param article_category_dict: Dictionary that maps articles to categories
        :return: New graph class with the nodes this time the categories, and the edges are links between categories
            (one link between categories implies that at least one article of that category has a link to the other
            category)
        """

        category_adj_list = defaultdict(set)
        for k, v in self._adj_list.items():
            category_adj_list[article_category_dict[k]].update(set([article_category_dict[node] for node in v]))

        return Graph.from_dict(category_adj_list)

    # This method will not be used any more. We can deduce the exact edges needed to be cut from the min_max algorithm
    def min_cut_kagler(self, source, sink, max_flow, iterations=20):
        """
        This method computes the exact edges that need to be removed in order to disconnect two pages. For this purpose
        will help ourselves with the max_flow approach (which gives the exact number of edges that need to be deleted
        for this to happen and also takes into account the fact that the graph is directed.
        This function is just an approximation, but given that we have the real result we will try to match it
        :param source: Article page we want to disconnect from sink
        :param sink: Article page we want to disconnect from source
        :param max_flow: Min Cut value
        :param iterations: Number of iterations before we stop looking for set of edges
        :return: Set of edges to delete
        """

        initial_edges = self.get_edges('dict')
        champion_set = []

        for iteration in range(iterations):
            edges = copy.deepcopy(initial_edges)
            while len(set(x for l in list(edges.values()) for x in l)) > 2:
                v1, v2 = self.get_random_edge(edges, source, sink)
                edges = self.contract_node(edges, v1, v2)
            edges_candidate = list({k: initial_edges[k] for k, v in edges.items()}.values())
            if iteration == 0:
                heapq.heappush(champion_set, (-len(edges), edges_candidate))
            else:
                heapq.heappushpop(champion_set, (-len(edges), edges_candidate))
            if len(edges) == max_flow:
                result = list(heapq.heappop(champion_set)[0])
                result[0] = -result[0]
                return heapq.heappop(champion_set)
            if iteration == iterations - 1:
                print('Max Flow and Kagler\'s algorithm do not converge')
                print('The minimum number of edges required to remove to disconnect is: ', max_flow)
                print('The best edges to do this are: ', heapq.heappop(champion_set)[1])

    @staticmethod
    def find_min_cut(initial_graph, residual_graph):
        """
        Following the max flow logic, we can compute the set of minimum cut egdes by taking those edges which initially had
        some kind of weight (different to 0), and after the max flow computation ended up having a weight of zero.

        :param initial_graph: Initial graph with the correct capacities
        :param residual_graph: Graph after obtaining the max flow, which we will compare against the first
        :param plot: Bool to plot the edges required to be cut
        :return: Set of edges to be cut
        """
        min_cut_edges = []
        for key in initial_graph:
            for sub_key in initial_graph[key]:
                if (initial_graph[key][sub_key] > 0) & (residual_graph[key][sub_key] == 0):
                    min_cut_edges.append([key, sub_key])
        return min_cut_edges



class WeightedGraph(Graph):

    def __init__(self):
        self._adj_list = defaultdict(dict)


    def add_edge(self, v, u, weight):
        """

        """
        self._adj_list[v][u] = weight
        self.add_vertex(u)


    def get_edges(self, output='tuple'):
        """
        Get the edges of the graph
        :param output: Output can be a list of tuples ('tuple'), a list of lists ('list') or a dictionary with lists
            ('dict')
        :return : list of (source, destination, weight) tuples
        """
        if output == 'tuple':
            return [(v, u, self._adj_list[v][u]) for v in self._adj_list.keys() for u in self._adj_list[v]]
        else:
            edge_list = [[v, u, self._adj_list[v][u]] for v in self._adj_list.keys() for u in self._adj_list[v]]
            if output == 'list':
                return edge_list
            else:
                return {i: edge for i, edge in enumerate(edge_list)}


    def nearest_neighbor(self, src):
        """
        Compute the shortest walk from src across all the vertices in the graph using the nearest neighbor heuristic

        :param src: start vertex

        :return : cost of the approximated minimum walk; return float('inf') if no walk is found
        """
        unvisited = set(self._adj_list.keys())
        cost = 0
        v = src
        while unvisited:
            unvisited.discard(v)
            min_ = [0, float('inf')]
            for u in self._adj_list[v].keys():
                if u in unvisited and self._adj_list[v][u] < min_[1]:
                    min_[0] = u
                    min_[1] = self._adj_list[v][u]
            if min_[1] == float('inf'):
                break
            cost += min_[1]
            v = min_[0]
        if unvisited:
            cost = float('inf')
        return cost


    # TODO: probably useless method
    def to_undirected(self):
        """
        Convert a directed graph into an undirected one.
        """
        wg = WeightedGraph()
        w = -10000
        for v in self._adj_list.keys():
            wg.add_edge(v, v*-1, w)
            wg.add_edge(v*-1, v, w)
        for v in self._adj_list.keys():
            for u in self._adj_list[v]:
                wg.add_edge(v*-1, u, self._adj_list[v][u])
                wg.add_edge(u, v*-1, self._adj_list[v][u])
        return wg


    # TODO: probably garbage
    def mst_prim(self, src):
        """
        Compute the Minimum spanning tree for an undirected graph.
        """
        processed = defaultdict(lambda: float('inf'))
        predecessors = {}
        dist = {}
        queue = []
        heapq.heappush(queue, (0, src, None))
        while queue:
            (d, u, pred) = heapq.heappop(queue)
            dist[u] = d
            predecessors[u] = pred
            for v in self._adj_list[u].keys():
                if v not in dist:
                    if dist[u] + self._adj_list[u][v] < processed[v]:
                        processed[v] = dist[u] + self._adj_list[u][v]
                        heapq.heappush(queue, (processed[v], v, u))
        return sum(dist.values())
