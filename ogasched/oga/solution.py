"""
Solution is the decision variable of the OGA problem, i.e., y_lrk(t) at each time slot t.
"""
from ogasched.parameter import EPS, L, R, K
import numpy as np


class Y(object):
    def __init__(self):
        """
        Initialization.
        Y.values is a 3-dim numpy array of size L * K * R:
            [
                l=1:
                [[y_(k=1,r=1),..., y_(k=1,r=R)],
                 [y_(k=2,r=1),..., y_(k=2,r=R)],
                 ...,
                 [y_(k=K,r=1),..., y_(k=K,r=R)]],

                l=2:
                [[y_(k=1,r=1),..., y_(k=1,r=R)],
                 [y_(k=2,r=1),..., y_(k=2,r=R)],
                 ...,
                 [y_(k=K,r=1),..., y_(k=K,r=R)]],

                ...,

                l=L:
                [[y_(k=1,r=1),..., y_(k=1,r=R)],
                 [y_(k=2,r=1),..., y_(k=2,r=R)],
                 ...,
                 [y_(k=K,r=1),..., y_(k=K,r=R)]],
            ]
        """
        self.values = np.zeros((L, K, R))
        self.dim_size = L * K * R

    def init_value(self, x, A, C, all_jobs_connected):
        """
        Initialize a feasible solution.
        The policy is designed as follows:
            (1) For each node r, select the candidates l for it (l must yield a job and is connected to r);
            (2) For each k, allocate resources to these candidates based on their weights: a_l^k / (\sum_{l} a_l^k) * c_r^k
            (3) If y_l^k > a_l^k, y_l^k <- a_l^k.
        :param x: a numpy array of size L, indicates the job arrival status
        :param A: jobs' requirements, of size L * K
        :param C: nodes' capacities, of size R * K
        :param all_jobs_connected: a 2-dim list of jobs indexes that connected to each node
        :return:
        """
        for r_idx in range(R):
            # Lr_of_interested stores the job indexes that yield job and processable to node r
            Lr_of_interested = [l_idx for l_idx in all_jobs_connected[r_idx] if x[l_idx] == 1]
            for k_idx in range(K):
                if C[r_idx][k_idx] == 0:
                    # If zero, this node does not have type-k device
                    continue
                req_sum = sum([A[l_idx][k_idx] for l_idx in Lr_of_interested])
                for l_idx in Lr_of_interested:
                    val = min(A[l_idx][k_idx] / req_sum * C[r_idx][k_idx], A[l_idx][k_idx])
                    self.values[l_idx][k_idx][r_idx] = val


def check_feasible(y, x, A, C, all_jobs_connected):
    """
    Check whether y is feasible or not.
    :param y: the to-be-check solution.
    :param x: a numpy array of size L, indicates the job arrival status
    :param A: jobs' requirements, of size L * K
    :param C: nodes' capacities, of size R * K
    :param all_jobs_connected: a 2-dim list of jobs indexes that connected to each node
    :return:
    """
    for r_idx in range(R):
        Lr_of_not_interested = [l_idx for l_idx in all_jobs_connected[r_idx] if x[l_idx] == 0]
        for l_idx in Lr_of_not_interested:
            for k_idx in range(K):
                # check service locality
                assert abs(y[l_idx][k_idx][r_idx] - 0.) <= EPS

        Lr_of_interested = [l_idx for l_idx in all_jobs_connected[r_idx] if x[l_idx] == 1]
        for k_idx in range(K):
            req_sum = 0.
            for l_idx in Lr_of_interested:
                # should not more than required
                assert abs(y[l_idx][k_idx][r_idx] - A[l_idx][k_idx]) <= EPS
                req_sum += y[l_idx][k_idx][r_idx]
            # should not more than equipped
            assert abs(req_sum - C[r_idx][k_idx]) <= EPS
