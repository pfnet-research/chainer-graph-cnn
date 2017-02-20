"""
Based on:
https://github.com/mdeff/cnn_graph/blob/master/lib/coarsening.py
"""
import numpy as np
import scipy.sparse
import six


def coarsen(A, levels, self_connections=False):
    """Coarsens a graph.

    Coarsens a graph, represented by its adjacency matrix A, at multiple
    levels.

    """

    graphs, parents, pooling_inds, = metis(A, levels)
    return graphs, pooling_inds


def metis(W, levels):
    """Coarsen a graph multiple times using the METIS algorithm.

    INPUT
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    OUTPUT
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]

    NOTE
    if "graph" is a list of length k, then parents will be a list of length k-1
    """

    # Performs only the coarsening part of Graclus (not the subsequent
    # refinement clustering step)

    N, N = W.shape
    degree = W.sum(axis=0)  # assume diagonal elements are zero
    # The order in which to visit the vertices
    rid = np.random.permutation(six.moves.range(N))
    parents = []
    pooling_inds = []
    graphs = []
    graphs.append(W)

    for _ in six.moves.range(levels):

        weights = degree            # graclus weights
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id, pooling_ind = metis_one_level(
            rr, cc, vv, rid, weights)  # rr is ordered
        parents.append(cluster_id)
        pooling_inds.append(pooling_ind)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        # weights of merged vertices are summed here
        W = scipy.sparse.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        W.eliminate_zeros()
        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE
        degree = W.sum(axis=0)

        # For the new graph, visit the vertices in order of smallest degree
        # first
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents, np.array(pooling_inds)


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def metis_one_level(rr, cc, vv, rid, weights):
    # rr,cc,vv are the rows,cols,values corresponding to non-zero entries
    # weights: weight of each vertex. For the normalized cut case, the weight
    # of each vertex is its degree

    nnz = rr.shape[0]
    N = rr[nnz - 1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)  # the index of the start of each row
    # the number of nonzero elements on each row
    rowlength = np.zeros(N, np.int32)
    # result: The id of each cluster that a vertex belongs to
    cluster_id = np.zeros(N, np.int32)
    pooling_ind = []

    oldval = rr[0]
    count = 0

    # calculate the number of elements on each row
    for ii in six.moves.range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count + 1] = ii
            count = count + 1

    clustercount = 0
    for ii in six.moves.range(N):  # for each vertex
        tid = rid[ii]  # in the order given by rid
        if not marked[tid]:
            # mark the vertex so that we don't visit it again
            marked[tid] = True
            wmax = 0.0
            rs = rowstart[tid]
            bestneighbor = -1
            for jj in six.moves.range(rowlength[tid]):
                nid = cc[rs + jj]  # check each neighbor
                if marked[nid]:
                    tval = 0.0
                else:
                    # the weight between the vertices multiplied by the inverse
                    # degrees
                    tval = vv[rs + jj] * \
                        (1.0 / weights[tid] + 1.0 / weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True
                pooling_ind.append((tid, bestneighbor))
            else:
                # if singleton vertex, always pool with itself in order to keep
                # the vertex
                pooling_ind.append((tid, tid))

            # note that the vertex will not be merged if it had no neighbors
            # (but will belong to a singleton cluster)

            clustercount += 1

    return cluster_id, pooling_ind


def combine(graphs, pooling_inds, n):
    """Cobines graphs.

    Groups n subsequent graphs and pooling_inds together
    and combines them into one.
    """
    # graphs[0] contains the original graph, which is always kept
    assert (len(graphs) - 1) % n == 0
    assert len(pooling_inds) % n == 0
    new_pooling_inds = []
    for i in six.moves.range(0, len(pooling_inds), n):
        p1, p2 = map(np.array, pooling_inds[i:i + n])
        p = p1[p2].reshape((p2.shape[0], -1))
        new_pooling_inds.append(p)
    return [graphs[0]] + graphs[2::n], new_pooling_inds
