

class Graph:
    
    def __init__(self):
        """
        Initialize the Graph object

        :attribute edges : dictionary of vertices (keys) to sets of vertices (values)
        """
        self.edges = {}


    def get_vertices(self):
        """
        Get the vertices of the graph

        :return : list of vertices
        """
        return self.edges.keys()

    
    def get_edges(self):
        """
        Get the edges of the graph

        :return : list of (source, destination) tuples
        """
        return [(v, u) for v in self.edges.keys() for u in self.edges[v]]


    def add_edge(self, v, u):
        """
        Add an edge to the graph

        :param v : source vertex
        :param u : destination vertex

        :return : 
        """
        try:
            self.edges[v].add(u)
        except:
            self.edges[v] = set([u,]) # set of vertices or list of vertices?

    
    def add_vertex(self, v): 
        """
        Add a vertex to the graph

        :param v : new vertex
        
        :return : 
        """
        if v not in self.edges:
            self.edges[v] = []


    def __repr__(self):
        """
        Represent the graph as a string of tuples
        """
        return str(self.get_edges())



g = Graph()

g.add_edge(1, 5)
g.add_edge(1, 3)
g.add_edge(1, 9)
g.add_edge(1, 2)
g.add_edge(3, 5)
g.add_edge(2, 5)
g.add_edge(2, 5)

g.add_vertex(0)

print(g)