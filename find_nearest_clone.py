#! /usr/bin/env python3

import os
import math
from collections import deque

# Complete the findShortest function below.

class GraphNode:
    def __init__(self, value, adjacent_nodes = None):
        self.value = value
        self.adjacent_nodes = adjacent_nodes or set()
    
    def add(self, node):
        self.adjacent_nodes.add(node)
    
    def __repr__(self):
        return "GraphNode(%s, %s)" % (
            self.value, 
            self.adjacent_nodes
        )

def build_graph(graph_values, graph_from, graph_to):
    nodes = tuple(
        GraphNode(graph_value) 
        for graph_value in graph_values
    )

    #build adjacency lists:
    for node_from, node_to in zip(graph_from, graph_to):
        #use 0 prefixed indexes:
        node_from -= 1
        node_to -= 1
        #populate adjacency lists:
        nodes[node_from].add(nodes[node_to])
        nodes[node_to].add(nodes[node_from])
    return nodes

def bucket_graph_nodes(nodes):
    #bucket nodes by color:
    node_buckets = {}
    for node in nodes: 
        if node.value in node_buckets:
            node_buckets[node.value].add(node)
        else:
            node_buckets[node.value] = { node }
    return node_buckets

def has_path(node_buckets, node_value):
    return node_value in node_buckets and len(node_buckets[node_value]) > 1

def find_shortest_path(node_buckets, search_value):
    shortest_path = math.inf
    visited = set()

    for node in node_buckets[search_value]:
        visited.add(node)
        queue = deque(((node, 0),)) #node / lvl
        while queue:
            selected_node, lvl = queue.popleft()
            for adj_node in selected_node.adjacent_nodes:
                if adj_node not in visited:
                    if adj_node.value == search_value and lvl < shortest_path:
                        #this will never override shortest path to a bigger value
                        #because I don't add nodes in the queue when lvl is higher
                        #then the current shortest path:
                        shortest_path = lvl + 1
                    elif (lvl + 1) < shortest_path:
                        #don't search further then an existing shortest path
                        visited.add(adj_node)
                        queue.append((adj_node, lvl + 1))
        visited.clear()
    return shortest_path

#
# For the weighted graph, <name>:
#
# 1. The number of nodes is <name>_nodes.
# 2. The number of edges is <name>_edges.
# 3. An edge exists between <name>_from[i] to <name>_to[i].
#
#
def find_shortest(
        graph_from, 
        graph_to, 
        graph_values, 
        search_value
    ):
    # solve here
    graph_nodes = build_graph(graph_values, graph_from, graph_to)
    node_buckets = bucket_graph_nodes(graph_nodes)

    if has_path(node_buckets, search_value):
        shortest_path = find_shortest_path(node_buckets, search_value) 
    else:
        shortest_path = -1
    return shortest_path

if __name__ == '__main__':
    print("Expected 3:")
    print(
        find_shortest(
            (1, 1, 1, 1, 2, 3, 3, 4, 5, 5, 6, 6,  6,  7,  9,  9, 10, 10, 11, 13, 14, 14, 15, 15, 16, 16),
            (2, 3, 4, 5, 6, 6, 7, 8, 9, 6, 7, 8, 10, 11, 11, 12, 11, 12, 12,  1, 13,  1, 14, 16, 12, 17),
            (1, 1, 2, 3, 4, 1, 1, 1, 1, 4, 1, 2, 4, 2, 3, 1, 4),
            3
        )
    )
    
    
    print("Expected 3:")
    print(
        find_shortest(
            (1, 1, 2, 3),
            (2, 3, 4, 5),
            (1, 2, 3, 3, 2),
            2
        )
    )
    
    print("Expected 1:")
    print(
        find_shortest(
            (4, 1, 1, 4),
            (3, 2, 3, 2),
            (1, 2, 1, 1),
            1
        )
    )
