import networkx as nx
import numpy as np
import igraph as ig
# data generating functions


def simulate_random_dag(
    d: int,
    degree: float,
    graph_type: str = 'erdos-renyi',
    w_range: tuple = (0.5, 2.0)) -> nx.DiGraph:
    """Simulate random DAG with some expected degree.
    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)
    Returns:
        G: weighted DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    elif graph_type == 'barabasi-albert':
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for ii in range(1, d):
            dest = np.random.choice(bag, size=m)
            for jj in dest:
                B[ii, jj] = 1
            bag.append(ii)
            bag.extend(dest)
    elif graph_type == 'full':  # ignore degree, only for experimental use
        B = np.tril(np.ones([d, d]), k=-1)
    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])

    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G


def simulate_nonlinear_sem(G: nx.DiGraph):
    nonlinear_functions = [np.cos, np.tanh, np.cosh, np.tanh]
    pass


def simulate_sem(G: nx.DiGraph,
                 n: int,
                 x_dims: int,
                 sem_type: str = 'linear-gauss',
                 linear_type: str = 'linear',
                 noise_scale: float = 1.0) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.
    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM
    Returns:
        X: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d

    for j in ordered_vertices:

        if isinstance(noise_scale, tuple):
            rnoise_scale = np.random.uniform(noise_scale[0], noise_scale[1])
        else:
            rnoise_scale = noise_scale
        parents = list(G.predecessors(j))
        if linear_type == 'linear':
            eta = X[:, parents, 0].dot(W[parents, j])
        elif linear_type == 'nonlinear_1':
            eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
        elif linear_type == 'nonlinear_2':
            eta = (X[:, parents, 0] + 0.5).dot(W[parents, j])
        else:
            raise ValueError('unknown linear data type')

        if sem_type == 'linear-gauss':
            if linear_type == 'linear':
                X[:, j, 0] = eta + np.random.normal(scale=rnoise_scale, size=n)
            elif linear_type == 'nonlinear_1':
                X[:, j, 0] = eta + np.random.normal(scale=rnoise_scale, size=n)
            elif linear_type == 'nonlinear_2':
                X[:, j, 0] = 2. * np.sin(eta) + eta + np.random.normal(
                    scale=rnoise_scale, size=n)
        elif sem_type == 'linear-exp':
            X[:, j,
              0] = eta + np.random.exponential(scale=rnoise_scale, size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j, 0] = eta + np.random.gumbel(scale=rnoise_scale, size=n)
        else:
            raise ValueError('unknown sem type')
    if x_dims > 1:
        for i in range(x_dims - 1):
            X[:, :, i + 1] = np.random.normal(
                scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(
                    scale=noise_scale, size=1) + np.random.normal(
                        scale=noise_scale, size=(n, d))
        X[:, :, 0] = np.random.normal(
            scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(
                scale=noise_scale, size=1) + np.random.normal(
                    scale=noise_scale, size=(n, d))
    return X
