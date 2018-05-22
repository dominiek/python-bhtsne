
import numpy as np
from bhtsne_wrapper import BHTSNE

class InvalidSeedPositionError(Exception):
    pass

def tsne(data, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, seed_positions=np.array([]), max_iter=1000):
    tsne = BHTSNE()
    skip_random_init = False
    if len(seed_positions) > 0:
        skip_random_init = True
        if seed_positions.shape[0] != data.shape[0]:
            raise InvalidSeedPositionError("Seed positions needs to be same number of rows as input matrix")
    Y = tsne.run(data, data.shape[0], data.shape[1], dimensions, perplexity, theta, rand_seed, seed_positions, skip_random_init, max_iter)
    return Y
