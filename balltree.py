# function build_ball_tree(data_points):
#     if len(data_points) < threshold:
#         return LeafNode(data_points)
#     else:
#         dimension = find_dimension_with_highest_variance(data_points)
#         median = find_median(data_points, dimension)
#         left_subset = [point for point in data_points if point[dimension] < median]
#         right_subset = [point for point in data_points if point[dimension] >= median]
#         return InternalNode(build_ball_tree(left_subset), build_ball_tree(right_subset))

# Import numpy for array operations
import numpy as np


# Define a class for Ball Tree nodes
class BallTreeNode:
    def __init__(self, data, centroid, radius):
        self.data = data  # The data points contained in the node
        self.centroid = centroid  # The centroid of the node
        self.radius = radius  # The radius of the node
        self.left = None  # The left child node
        self.right = None  # The right child node


# Define a function to compute the distance between two points
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# Define a function to build a Ball Tree from a set of data points
def build_ball_tree(data, leaf_size=20):
    # If the data is small enough, create a leaf node
    if len(data) <= leaf_size:
        centroid = np.mean(data, axis=0)  # Compute the centroid of the data
        radius = max(distance(p, centroid) for p in data)  # Compute the radius of the data
        return BallTreeNode(data, centroid, radius)
    else:
        # Find the dimension with the highest variance
        variances = np.var(data, axis=0)
        dimension = np.argmax(variances)

        # Split the data at the median of the selected dimension
        median = np.median(data[:, dimension])
        left_data = data[data[:, dimension] < median]
        right_data = data[data[:, dimension] >= median]

        # Recursively build the left and right child nodes
        left_node = build_ball_tree(left_data, leaf_size)
        right_node = build_ball_tree(right_data, leaf_size)

        # Compute the centroid and radius of the parent node
        centroid = (left_node.centroid * len(left_node.data) + right_node.centroid * len(right_node.data)) / len(data)
        radius = max(distance(p, centroid) for p in data)

        # Create the parent node and link it to the child nodes
        node = BallTreeNode(data, centroid, radius)
        node.left = left_node
        node.right = right_node

        # Return the parent node
        return node


def query_ball_tree(node, point, k):
    def sort_by_distance():
        # sort the data points in the node by distance to the query point
        distances = np.array([distance(p, point) for p in node.data])
        indices = np.argsort(distances)
        return node.data[indices][:k], distances[indices][:k]

    # If the node is a leaf node, return the data points it contains
    if node.left is None and node.right is None:
        return sort_by_distance()
    else:
        # If the node is not a leaf node, check if the query point is within the radius of the node
        if distance(point, node.centroid) <= node.radius:
            # If the query point is within the radius of the node, recursively query the child nodes
            left_points, left_distance = query_ball_tree(node.left, point, k)
            right_points, right_distance = query_ball_tree(node.right, point, k)

            # Return the k nearest neighbors from the union of the child nodes
            return np.concatenate((left_points, right_points))[:k], np.concatenate((left_distance, right_distance))[:k]
        else:
            # If the query point is not within the radius of the node, return the data points contained in the node
            # sorted by distance to the query point
            return sort_by_distance()


def main():
    # data = np.array([[2, 3], [5, 4], [9, 6], [8, 1], [7, 2], [6, 3], [1, 5], [4, 7], [3, 9]])
    data = np.array([[2, 3], [5, 4], [9, 6], [8, 1]])
    tree = build_ball_tree(data, leaf_size=2)
    result = query_ball_tree(tree, np.array([[5, 5]]), 1)
    print(result)


if __name__ == '__main__':
    main()
