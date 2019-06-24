#! /usr/bin/env python3

# PRACTICE: SEGMENT TREES
# I usually practice by writing multiple slightly different versions 
# of the same algorithm / technique multiple times to get it to muscle 
# memory
# 
# min / sum range query practice: dynamic programming, min segment trees, sum segment trees

import math

# dynamic programming solution for min within a range:

def build_min_matrix(array):
    array_length = len(array)
    matrix = [
        [array[i]]*array_length for i in range(array_length)
    ]
    for span in range(1, array_length):
        i, j = 0, span
        while j < array_length:
            matrix[i][j] = min(matrix[i][j - 1], matrix[i + 1][j])
            i += 1
            j += 1
    return matrix

# dynamic programming solution for sum within a range:

def build_sum_matrix(array):
    array_length = len(array)

    def build_row(index):
        row = [0]*array_length
        row[index] = array[index]
        return row

    matrix = [
        build_row(i) for i in range(array_length)
    ]
    for span in range(1, array_length):
        i, j = 0, span
        while j < array_length:
            matrix[i][j] = matrix[i][j - 1] + matrix[i + 1][j] - matrix[i + 1][j - 1]
            i += 1
            j += 1
    return matrix

# segment tree for min within a range:
class SegmentTree:
    tree = None
    _high = None

    def __init__(self, array = None):
        if array:
            self.build_recursive(array)

    def get_tree_size(number_of_elements):
        return 2**(len(bin(number_of_elements)) - 1)

    def build_recursive(self, array):
        raise NotImplementedError("This method is not implemented")

    def build_iterative(self, array):
        raise NotImplementedError("This method is not implemented")

    def search_recursive(self, query_low_idx, query_high_idx):
        raise NotImplementedError("This method is not implemented")

    def search_iterative(self, query_low_idx, query_high_idx):
        raise NotImplementedError("Build recursive is not implemented")

class MinSegmentTree(SegmentTree):

    def build_recursive(self, array):
        array_length = len(array)
        self.tree = [math.inf]*SegmentTree.get_tree_size(array_length)
        self._high = array_length - 1
        self._build_recursive(array, 0, self._high) 

    def _build_recursive(self, array, low_idx, high_idx, node_idx = 0):
        if low_idx == high_idx:
            self.tree[node_idx] = array[low_idx]
            return array[low_idx]

        mid_idx = (low_idx + high_idx) // 2

        self.tree[node_idx] = min(
            self._build_recursive(array, low_idx, mid_idx, node_idx*2 + 1),
            self._build_recursive(array, mid_idx + 1, high_idx, node_idx*2 + 2)
        )
        return self.tree[node_idx]

    def build_iterative(self, array):
        array_length = len(array)

        # initialize tree and high:
        self.tree = [math.inf]*SegmentTree.get_tree_size(array_length)
        self._high = array_length - 1

        stack = (0, self._high, 0, None)
        
        def is_visited(node_idx):
            return self.tree[node_idx*2 + 1] != math.inf or self.tree[node_idx*2 + 2] != math.inf

        while stack:
            low_idx, high_idx, node_idx = stack[:-1] 

            if low_idx == high_idx:
                self.tree[node_idx] = array[low_idx]
                stack = stack[-1]
            else:
                mid_idx = (low_idx + high_idx) // 2
                if is_visited(node_idx):
                    # visited:
                    self.tree[node_idx] = min(
                        self.tree[node_idx*2 + 1], 
                        self.tree[node_idx*2 + 2]
                    )
                    stack = stack[-1]
                else:
                    # not visited, push to stack:
                    stack = (
                        low_idx, mid_idx, node_idx*2 + 1, 
                        (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                    )

    def search_recursive(self, low_idx, high_idx):
        return self._search_recursive(low_idx, high_idx, 0, self._high)

    def _search_recursive(self, 
            query_low_idx, query_high_idx, low_idx, high_idx, node_idx = 0):

        if query_low_idx <= low_idx and query_high_idx >= high_idx:
            # full overlap, return the value
            min_value = self.tree[node_idx]
        elif query_low_idx > high_idx or query_high_idx < low_idx:
            # no overlap
            min_value = math.inf
        else:
            mid_idx = (low_idx + high_idx) // 2
            min_value = min(
                self._search_recursive(
                    query_low_idx, query_high_idx, low_idx, mid_idx, node_idx*2 + 1),
                self._search_recursive(
                    query_low_idx, query_high_idx, mid_idx + 1, high_idx, node_idx*2 + 2)
            )
        return min_value

    def search_iterative(self, query_low_idx, query_high_idx):
        min_value = math.inf
        stack = (0, self._high, 0, None)

        while stack:
            low_idx, high_idx, node_idx, stack = stack

            if query_low_idx <= low_idx and query_high_idx >= high_idx:
                #overlap
                min_value = min(min_value, self.tree[node_idx])
            elif query_low_idx > high_idx or query_high_idx < low_idx:
                #no overlap or visited already
                pass
            else:
                mid_idx = (low_idx + high_idx) // 2
                stack = (low_idx, mid_idx, node_idx*2 + 1, 
                    (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                )
        return min_value

class LazyPropagationMinSegmentTree(MinSegmentTree):

    def build_recursive(self, array):
        super().build_recursive(array)
        self.lazy_tree = [0]*len(self.tree)

    def build_iterative(self, array):
        super().build_iterative(array)
        self.lazy_tree = [0]*len(self.tree)

    def update_recursive(self, query_low_idx, query_high_idx, value):
        self._update_recursive(
            query_low_idx, query_high_idx, 0, self._high, 0, value)

    def _update_recursive(self, query_low_idx, query_high_idx, low_idx, high_idx, node_idx, value):
        #propagation:
        if self.lazy_tree[node_idx]:
            self._propagade_lazy_tree(node_idx, low_idx != high_idx)

        if query_low_idx <= low_idx and query_high_idx >= high_idx:
            # overlap:
            self.tree[node_idx] += value
            if low_idx != high_idx:
                #propagade down the lazy tree:
                self.lazy_tree[node_idx*2 + 1] += value
                self.lazy_tree[node_idx*2 + 2] += value
            result = self.tree[node_idx]
        elif query_low_idx > high_idx or query_high_idx < low_idx:
            # no overlap, do not update, return the current value for the parent min function to compare
            result = self.tree[node_idx]
        else:
            mid_idx = (low_idx + high_idx) // 2
            result = min(
                self._update_recursive(query_low_idx, query_high_idx, low_idx, mid_idx, node_idx*2 + 1, value),
                self._update_recursive(query_low_idx, query_high_idx, mid_idx + 1, high_idx, node_idx*2 + 2, value)
            )
            self.tree[node_idx] = result
        return result

    def update_iterative(self, query_low_idx, query_high_idx, value):

        stack = (0, self._high, 0, None)
        visited = set()

        while stack:
            low_idx, high_idx, node_idx = stack[:-1]

            #propagade lazy tree if node exists
            if self.lazy_tree[node_idx]:
                self._propagade_lazy_tree(node_idx, low_idx != high_idx)

            if query_low_idx <= low_idx and query_high_idx >= high_idx:
                self.tree[node_idx] += value

                if low_idx != high_idx:
                    self.lazy_tree[node_idx*2 + 1] += value
                    self.lazy_tree[node_idx*2 + 2] += value

                stack = stack[-1] # pop stack
            elif query_low_idx > high_idx or query_high_idx < low_idx:
                stack = stack[-1] # pop stack
            elif node_idx not in visited:
                visited.add(node_idx)

                mid_idx = (low_idx + high_idx) // 2

                #push left and right half to stack:
                stack = (low_idx, mid_idx, node_idx*2 + 1, 
                    (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                )
            else:
                self.tree[node_idx] = min(
                    self.tree[node_idx*2 + 1], self.tree[node_idx*2 + 2]
                )
                stack = stack[-1] # pop stack

    def _propagade_lazy_tree(self, node_idx, has_children):
        lazy_value = self.lazy_tree[node_idx]
        self.lazy_tree[node_idx] = 0
        self.tree[node_idx] += lazy_value
        if has_children:
            self.lazy_tree[node_idx*2 + 1] += lazy_value
            self.lazy_tree[node_idx*2 + 2] += lazy_value

    def _search_recursive(self, 
            query_low_idx, query_high_idx, low_idx, high_idx, node_idx = 0):

        # propagation
        if self.lazy_tree[node_idx]:
            self._propagade_lazy_tree(node_idx, low_idx != high_idx)

        if query_low_idx <= low_idx and query_high_idx >= high_idx:
            # overlap:
            result = self.tree[node_idx]
        elif query_low_idx > high_idx or query_high_idx < low_idx:
            # no overlap:
            return math.inf
        else:
            mid_idx = (low_idx + high_idx) // 2
            result = min(
                self._search_recursive(query_low_idx, query_high_idx, low_idx, mid_idx, node_idx*2 + 1),
                self._search_recursive(query_low_idx, query_high_idx, mid_idx + 1, high_idx, node_idx*2 + 2)
            )
        return result

    def search_iterative(self, query_low_idx, query_high_idx):

        stack = (0, self._high, 0, None)
        min_value = math.inf

        while stack:
            low_idx, high_idx, node_idx, stack = stack

            if self.lazy_tree[node_idx]:
                self._propagade_lazy_tree(node_idx, low_idx != high_idx)

            if query_low_idx <= low_idx and query_high_idx >= high_idx:
                min_value = min(min_value, self.tree[node_idx])
            elif query_low_idx > high_idx or query_high_idx < low_idx:
                pass
            else:
                mid_idx = (low_idx + high_idx) // 2
                stack = (low_idx, mid_idx, node_idx*2 + 1,
                    (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                )
        return min_value

# segment tree for sum within a range:

class SumSegmentTree(SegmentTree):

    def build_recursive(self, array):
        array_length = len(array)
        self.tree = [0]*SegmentTree.get_tree_size(array_length)
        self._high = array_length - 1
        self._build_recursive(array, 0, self._high)

    def _build_recursive(self, array, low_idx, high_idx, node_idx = 0):
        if low_idx == high_idx:
            self.tree[node_idx] = array[low_idx]
            return array[low_idx]

        mid_idx = (low_idx + high_idx) // 2
        self.tree[node_idx] = (
            self._build_recursive(array, low_idx, mid_idx, node_idx*2 + 1) + 
            self._build_recursive(array, mid_idx + 1, high_idx, node_idx*2 + 2)
        )
        return self.tree[node_idx]

    def build_iterative(self, array):
        array_length = len(array)

        self.tree = [0]*SegmentTree.get_tree_size(array_length)
        self._high = array_length - 1

        stack = (0, self._high, 0, None)

        def is_visited(node_idx):
            return bool(
                self.tree[node_idx*2 + 1] + self.tree[node_idx*2 + 2]
            )
        
        while stack:
            low_idx, high_idx, node_idx = stack[:-1]

            if low_idx == high_idx:
                self.tree[node_idx] = array[low_idx]
                stack = stack[-1]
            elif is_visited(node_idx):
                self.tree[node_idx] = self.tree[node_idx*2 + 1] + self.tree[node_idx*2 +2]
                stack = stack[-1]
            else:
                mid_idx = (low_idx + high_idx) // 2
                stack = (
                    low_idx, mid_idx, node_idx*2 + 1, (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                )

    def search_recursive(self, query_low_idx, query_high_idx):
        return self._search_recursive(query_low_idx, query_high_idx, 0, self._high)

    def _search_recursive(self, query_low_idx, query_high_idx, low_idx, high_idx, node_idx = 0):
        if query_low_idx <= low_idx and query_high_idx >= high_idx:
            # overlap found
            result = self.tree[node_idx]
        elif query_low_idx > high_idx or query_high_idx < low_idx:
            # no overlap
            result = 0
        else:
            mid_idx = (low_idx + high_idx) // 2
            result = (
                self._search_recursive(query_low_idx, query_high_idx, low_idx, mid_idx, node_idx*2 + 1) +
                self._search_recursive(query_low_idx, query_high_idx, mid_idx + 1, high_idx, node_idx*2 + 2)
            )
        return result

    def search_iterative(self, query_low_idx, query_high_idx):
        stack = (0, self._high, 0, None)

        result_sum = 0

        while stack:
            low_idx, high_idx, node_idx, stack = stack

            if query_low_idx <= low_idx and query_high_idx >= high_idx:
                #total overlap, add to sum
                result_sum += self.tree[node_idx]
            elif query_low_idx > high_idx or query_high_idx < low_idx:
                pass
            else:
                mid_idx = (low_idx + high_idx) // 2
                stack = (low_idx, mid_idx, node_idx*2 + 1,
                    (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                )
        return result_sum

class LazyPropagationSumSegmentTree(SumSegmentTree):

    def build_recursive(self, array):
        super().build_recursive(array)
        self.lazy_tree = [0]*len(self.tree)

    def build_iterative(self, array):
        super().build_iterative(array)
        self.lazy_tree = [0]*len(self.tree)

    def update_recursive(self, query_low_idx, query_high_idx, value):
        self._update_recursive(query_low_idx, query_high_idx, 0, self._high, 0, value)

    def _update_recursive(self, query_low_idx, query_high_idx, low_idx, high_idx, node_idx, value):

        if self.lazy_tree[node_idx]:
            child_count = high_idx != low_idx and high_idx - low_idx + 1
            self._propagade_lazy_tree(node_idx, child_count)

        if query_low_idx <= low_idx and query_high_idx >= high_idx:
            # full overlap
            self.tree[node_idx] += value*(high_idx - low_idx + 1)

            if low_idx != high_idx:
                self.lazy_tree[node_idx*2 + 1] += value
                self.lazy_tree[node_idx*2 + 2] += value
            result = self.tree[node_idx]
        elif query_low_idx > high_idx or query_high_idx < low_idx:
            result = self.tree[node_idx]
        else:
            mid_idx = (low_idx + high_idx) // 2
            result = (
                self._update_recursive(query_low_idx, query_high_idx, low_idx, mid_idx, node_idx*2 + 1, value) +
                self._update_recursive(query_low_idx, query_high_idx, mid_idx + 1, high_idx, node_idx*2 + 2, value)
            )
            self.tree[node_idx] = result
        return result

    def update_iterative(self, query_low_idx, query_high_idx, value):
        stack = (0, self._high, 0, None) 
        visited = set()

        while stack:

            low_idx, high_idx, node_idx = stack[:-1]
            
            if self.lazy_tree[node_idx]:
                child_count = high_idx != low_idx and high_idx - low_idx + 1
                self._propagade_lazy_tree(node_idx, child_count) 

            if query_low_idx <= low_idx and query_high_idx >= high_idx:
                #overlap
                self.tree[node_idx] += value*(high_idx - low_idx + 1)
                if low_idx != high_idx:
                    self.lazy_tree[node_idx*2 + 1] += value
                    self.lazy_tree[node_idx*2 + 2] += value
                stack = stack[-1]
            elif query_low_idx > high_idx or query_high_idx < low_idx:
                stack = stack[-1]
            elif node_idx not in visited:
                visited.add(node_idx)
                mid_idx = (low_idx + high_idx) // 2
                stack = (low_idx, mid_idx, node_idx*2 + 1,
                    (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                )
            else:
                self.tree[node_idx] = self.tree[node_idx*2 + 1] + self.tree[node_idx*2 + 2]
                stack = stack[-1]

    def _search_recursive(self, query_low_idx, query_high_idx, low_idx, high_idx, node_idx = 0):

        if self.lazy_tree[node_idx]:
            child_count = high_idx != low_idx and high_idx - low_idx + 1
            self._propagade_lazy_tree(node_idx, child_count) 

        if query_low_idx <= low_idx and query_high_idx >= high_idx:
            result = self.tree[node_idx]
        elif query_low_idx > high_idx or query_high_idx < low_idx:
            result = 0
        else:
            mid_idx = (low_idx + high_idx) // 2
            result = (
                self._search_recursive(query_low_idx, query_high_idx, low_idx, mid_idx, node_idx*2 + 1) +
                self._search_recursive(query_low_idx, query_high_idx, mid_idx + 1, high_idx, node_idx*2 + 2)
            )
        return result

    def search_iterative(self, query_low_idx, query_high_idx):
        stack = (0, self._high, 0, None)
        total_sum = 0

        while stack:
            low_idx, high_idx, node_idx, stack = stack

            if self.lazy_tree[node_idx]:
                child_count = high_idx != low_idx and high_idx - low_idx + 1
                self._propagade_lazy_tree(node_idx, child_count) 

            if query_low_idx <= low_idx and query_high_idx >= high_idx:
                # overlap:
                total_sum += self.tree[node_idx]
            elif query_low_idx > high_idx or query_high_idx < low_idx:
                pass
            else:
                mid_idx = (low_idx + high_idx) // 2 
                stack = (low_idx, mid_idx, node_idx*2 + 1,
                    (mid_idx + 1, high_idx, node_idx*2 + 2, stack)
                )
        return total_sum

    def _propagade_lazy_tree(self, node_idx, child_count):
        lazy_value = self.lazy_tree[node_idx]
        self.lazy_tree[node_idx] = 0
        self.tree[node_idx] += lazy_value*(child_count or 1)

        if child_count:
            self.lazy_tree[node_idx*2 + 1] += lazy_value
            self.lazy_tree[node_idx*2 + 2] += lazy_value

if __name__ == "__main__":
    test = [-1, 3, 4, 0, 2, 1]

    print("TEST MATRIX") 
    print("Build min matrix test: %s" % test)
    min_matrix = build_min_matrix(test)
    for row in min_matrix:
        print(row)

    print()
    print("Build sum matrix test: %s" % test)
    sum_matrix = build_sum_matrix(test)
    for row in sum_matrix:
        print(row)

    print()
    print("Build min segment tree test: %s" % test)
    segment_tree = MinSegmentTree()
    segment_tree.build_recursive(test)
    print(segment_tree.tree)
    segment_tree.build_iterative(test)
    print(segment_tree.tree)
    print("TEST MIN SEGMENT TREE SEARCH [matrix == segment tree recursive == segment tree iterative]:")
    for low, high in (
            (0, 1), 
            (0, 3), 
            (1, 2), 
            (1, 4), 
            (1, 5)
        ):
        print(
            "[%s %s] %s == %s == %s" % (
                low, high, 
                min_matrix[low][high], 
                segment_tree.search_recursive(low, high),
                segment_tree.search_iterative(low, high),
            )
        )

    print()
    print("TEST MIN LAZY PROPAGATION SEGMENT TREE SEARCH:")
    segment_tree = LazyPropagationMinSegmentTree()
    segment_tree2 = LazyPropagationMinSegmentTree()
    segment_tree.build_recursive(test)
    segment_tree2.build_iterative(test)
    updated_test = test.copy()

    for low, high, value in (
            (0, 0, -1),
            (1, 2, 2),
            (2, 5, -1)
        ):
        segment_tree.update_recursive(low, high, value)
        segment_tree2.update_iterative(low, high, value)

        #update matrix test
        while low <= high:
            updated_test[low] += value
            low += 1

    #the result is this array
    min_matrix = build_min_matrix(updated_test)

    for low, high in (
            (0, 1), 
            (0, 3), 
            (1, 2), 
            (1, 4), 
            (1, 5)
        ):
        print(
            "[%s %s] %s == %s == %s" % (
                low, high, 
                min_matrix[low][high],
                segment_tree.search_recursive(low, high),
                segment_tree2.search_iterative(low, high),
            )
        )

    print()
    print("Build sum segment tree test: %s" % test)
    segment_tree = SumSegmentTree()
    segment_tree.build_recursive(test)
    print(segment_tree.tree)
    segment_tree.build_iterative(test)
    print(segment_tree.tree)
    print("TEST SUM SEGMENT TREE SEARCH [matrix == segment tree recursive == segment tree iterative]:")
    for low, high in (
            (0, 1), 
            (0, 3), 
            (1, 2), 
            (1, 4), 
            (1, 5),
            (0, 5)
        ):
        print(
            "[%s %s] %s == %s == %s:" % (
                low, high, 
                sum_matrix[low][high], 
                segment_tree.search_recursive(low, high),
                segment_tree.search_iterative(low, high)
            )
        )

    print()
    print("TEST SUM LAZY PROPAGATION SEGMENT TREE SEARCH:")
    segment_tree = LazyPropagationSumSegmentTree()
    segment_tree2 = LazyPropagationSumSegmentTree()
    segment_tree.build_recursive(test)
    segment_tree2.build_iterative(test)
    updated_test = test.copy()

    for low, high, value in (
            (0, 0, -1),
            (1, 2, 2),
            (2, 5, -1)
        ):
        segment_tree.update_recursive(low, high, value)
        segment_tree2.update_iterative(low, high, value)

        #update matrix test
        while low <= high:
            updated_test[low] += value
            low += 1
            
    sum_matrix = build_sum_matrix(updated_test)
    for low, high in (
            (0, 1), 
            (0, 3), 
            (1, 2), 
            (1, 4), 
            (1, 5),
            (0, 5),
        ):
        print(
            "[%s %s] %s == %s == %s" % (
                low, high, 
                sum_matrix[low][high],
                segment_tree.search_recursive(low, high),
                segment_tree2.search_iterative(low, high),
            )
        )
