import numpy as np
from scipy.spatial.distance import cdist


def compute_molecular_distances(atom_positions):
    distances = cdist(atom_positions, atom_positions)
    return distances


def ensemble_rmsd_matrix(conformations):
    n_conformations = conformations.shape[0]
    rmsd_matrix = np.zeros((n_conformations, n_conformations))

    for i in range(n_conformations):
        for j in range(i + 1, n_conformations):
            rmsd = np.sqrt(np.mean((conformations[i] - conformations[j]) ** 2))
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    return rmsd_matrix


def graph_shortest_paths(adjacency_matrix):
    n_vertices = adjacency_matrix.shape[0]
    dist = np.copy(adjacency_matrix)
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)

    for k in range(n_vertices):
        for i in range(n_vertices):
            for j in range(n_vertices):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


def compute_kernel_gram_matrix(X, gamma=1.0):
    pairwise_sq_dists = cdist(X, X, metric="sqeuclidean")
    K = np.exp(-gamma * pairwise_sq_dists)
    return K


n_atoms = 150
atom_coords = np.random.randn(n_atoms, 3)
mol_dists = compute_molecular_distances(atom_coords)

n_conformations = 50
conformations = np.random.randn(n_conformations, n_atoms, 3)
rmsd = ensemble_rmsd_matrix(conformations)

n_vertices = 80
adj = np.random.randint(0, 2, size=(n_vertices, n_vertices))
shortest = graph_shortest_paths(adj)

X = np.random.randn(200, 64)
K = compute_kernel_gram_matrix(X)
