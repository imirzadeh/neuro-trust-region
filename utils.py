import random
from pyDOE2 import lhs, ccdesign
import numpy as np

def normalize(v):
    """
    Make a unit vector out of `v`. Simply dividing by its norm. 
    """
    norm = np.linalg.norm(v, ord=2)
    return v/norm

def generate_samples(num_samples_inside, num_samples_boundary, x_k, delta_k):
    """
    Generates samples from both `inside the region` and `on the boundary`.
    To sample from the boundary, the *Central Composite* is used.
    To sample from the region interior, the  *Latin Hypercube Sampling* is used.

    Args:
        num_samples_inside (int): Number of desired samples inside the region.
        num_samples_boundary (int): Number of desired samples on the region boundary.
        x_k (vector): Center of the boundary.
        delta_k (float): Radius of the boundary.

    Returns:
        total_samples: list of new samples form the boundary
    """

    # 1. Determine the dimension of x_k: used for knowing the random vector dimensions
    x_k_dim = x_k.shape[0]

    # 2. Sample from boundary: Central Composite
    if x_k_dim == 1:
        samples_boundary = np.array([[-1.0], [1.0]])
    else:
        res = [normalize(v) for v in ccdesign(x_k_dim, center=[0 for i in range(x_k_dim)])]
        samples_boundary = np.array(res)

    samples_boundary = random.choices(delta_k * samples_boundary + x_k, k=num_samples_boundary)

    # 3. Sample from region: Latin Hypercube Sampling
    samples_inside = lhs(x_k_dim, samples=num_samples_inside)
    scaled_samples_inside = np.vectorize(lambda x: -delta_k +  x * delta_k * 2)(samples_inside)
    samples_inside = scaled_samples_inside + x_k

    # 4. Add directions to the center point x_k to get new candidate points
    total_samples = np.concatenate((np.concatenate((samples_boundary, samples_inside)),[[0.0 for i in range(x_k_dim)]]))
    for i in range(5):
        total_samples = np.concatenate((total_samples, [[0.0 for i in range(x_k_dim)]]))
    return total_samples


# if __name__ == "__main__":
#     x_k = np.array([0.0], dtype=np.float32)
#     delta_k = 2.0
#     print(generate_samples(5, 3, x_k, delta_k))