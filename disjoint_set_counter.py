#! /usr/bin/env python3

# HACKERRANK
# You will be q given queries. 
# After each query, you need to report the size of the largest friend circle (the largest group of friends) formed after considering that query. 

class DisjointSetCounterNode:
    def __init__(self, value, next_node = None):
        self.value = value
        self.next_node = next_node

    @property
    def root(self):
        selected_node = self
        next_selected_node = selected_node.next_node
        while next_selected_node:
            # while iterating to the root, collapse the nodes to the root:
            if next_selected_node.next_node:
                selected_node.next_node = next_selected_node.next_node 
            selected_node = next_selected_node
            next_selected_node = next_selected_node.next_node
        return selected_node
    
class DisjointSetCounter:
    # data structure can be visualized as follows:
    # value_dict = {
    #      1  : counter1
    #      2  : counter1
    #      3  : counter2 
    #      4  : counter2
    #      5  : counter3
    #      99 : counter1 
    #      76 : counter1 
    #      34 : counter1
    #      22 : counter2 
    #      8  : counter3 
    #      9  : counter3 
    # }
    # 
    # counters are linked to each other when 2 sets are merged, for example:
    # new query: (1, 3)
    # need to join counter1 and counter2 
    # a new counter will be created as follows: 
    # counter4, such that:
    # 
    #    counter1  counter2 
    #       \         / 
    #        \       / 
    #         \     / 
    #          \   / 
    #         counter4
    # 
    # etc. 
    # There is an additional optimization added when iterating through counters,
    # that is when root is called, I am collapsing the nodes so the tree is smaller
    def __init__(self):
        self.value_dict = dict()
        self.max_count = 0
    
    def __contains__(self, value):
        return value in self.value_dict
    
    def _update_max_count(self, candidate_count):
        if candidate_count > self.max_count:
            self.max_count = candidate_count
    
    def union(self, value_a, value_b):
        if value_a in self and value_b not in self:
            a_root = self.value_dict[value_b] = self.value_dict[value_a].root
            a_root.value += 1
            total = a_root.value
        elif value_a not in self and value_b in self:
            b_root = self.value_dict[value_a] = self.value_dict[value_b].root
            b_root.value += 1
            total = b_root.value
        elif value_a in self and value_b in self:
            a_root = self.value_dict[value_a].root
            b_root = self.value_dict[value_b].root
            if a_root != b_root:
                total = a_root.value + b_root.value
                a_root.next_node = b_root.next_node = DisjointSetCounterNode(total)
            else:
                total = 0
        else:
            total = 2 
            self.value_dict[value_a] = self.value_dict[value_b] = DisjointSetCounterNode(total)
        self._update_max_count(total)

# Complete the maxCircle function below.
def max_circle(queries):
    max_circles = [0]*len(queries)
    ds = DisjointSetCounter()
    for idx in range(len(queries)):
        friend_a, friend_b = queries[idx]
        ds.union(friend_a, friend_b)
        max_circles[idx] = ds.max_count
    return max_circles


if __name__ == '__main__':
    print(max_circle(
            (
                (1, 2),
                (3, 4),
                (1, 3),
                (5, 7),
                (5, 6),
                (7, 4),
            )
        )
    )
