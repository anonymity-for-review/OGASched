"""
Dominant Resource Fairness.
"""
import numpy as np
from ogasched.parameter import L, R, K, T
from ogasched import reward
from ogasched.oga.solution import check_feasible
from ogasched.utils.logging import logging
from ogasched.utils.utils import get_run_time


class DRF(object):
    def __init__(self, env):
        """
        Initialization.
        :param env: a Env instance
        """
        self.env = env

        # A is of size L * K
        self.A = np.array([job_type.a for job_type in self.env.job_types])

        # C is of size R * K
        self.C = np.array([node.c for node in self.env.nodes])

        self.all_nodes_connected = [job_type.nodes_connected for job_type in self.env.job_types]  # of length L
        self.all_jobs_connected = [node.jobs_connected for node in self.env.nodes]  # of length R

        self.all_utilities = [["linear"] * R] * K  # of size K * R
        for r_idx in range(len(self.env.nodes)):
            node = self.env.nodes[r_idx]
            for k_idx in range(K):
                self.all_utilities[k_idx][r_idx] = node.utilities[k_idx]

        # job arrivals of size T * L
        self.xs = self.env.X
        # solutions at each time t
        self.ys = np.zeros((T, L, K, R))
        # rewards at each time t
        self.rs = np.zeros(T)
        self.acc_r = 0.

    @get_run_time
    def run(self):
        """
        DRF allocates resources to jobs in the ascending order of their dominant shares.
        :return:
        """
        for t in range(0, T):
            x = self.xs[t]

            # U is the remaining resource, of size R * K
            U = self.C.copy()

            # vec_C is total
            vec_C = np.zeros((L, K))
            for l_idx in range(L):
                vec_C[l_idx, :] = np.array(
                    [sum(self.C[r_idx][k_idx] for r_idx in self.env.job_types[l_idx].nodes_connected)
                     for k_idx in range(K)]
                )
            # vec_U is remaining
            vec_U = vec_C.copy()

            # calculate the dominant share of each job type
            # s is of size L
            s = np.array(
                [(max([self.A[l_idx][k_idx] / vec_C[l_idx][k_idx]] for k_idx in range(K))) for l_idx in range(L)]
            )
            sorted_s_indexes = np.argsort(s)

            # choose the port with the ascending order of dominant share
            for l_idx in sorted_s_indexes:
                can_be_allocated = [min(request, remained) for (request, remained) in zip(self.A[l_idx], vec_U[l_idx])]
                for k_idx in range(K):
                    has_allocated = 0.
                    for r_idx in self.env.job_types[l_idx].nodes_connected:
                        tmp = min(self.A[l_idx][k_idx], U[r_idx][k_idx])
                        if has_allocated + tmp <= can_be_allocated[k_idx]:
                            self.ys[t][l_idx][k_idx][r_idx] = tmp
                            U[r_idx][k_idx] -= tmp
                        else:
                            self.ys[t][l_idx][k_idx][r_idx] = can_be_allocated[k_idx] - has_allocated
                            U[r_idx][k_idx] -= self.ys[t][l_idx][k_idx][r_idx]
                        has_allocated += self.ys[t][l_idx][k_idx][r_idx]

                # update vec_U
                for inner_l in range(L):
                    vec_C[l_idx, :] = np.array(
                        [sum(U[r_idx][k_idx] for r_idx in self.env.job_types[inner_l].nodes_connected)
                         for k_idx in range(K)]
                    )

            check_feasible(self.ys[t], x, self.A, self.C, self.all_jobs_connected)
            re = reward.reward(x, self.ys[t], self.all_nodes_connected, utilities=self.all_utilities)
            self.rs[t] = re
            self.acc_r += re
            logging.info("[DRF] Reward at t = %d: %.2f", t, self.rs[t])

        logging.info("[DRF] Acc. Reward within T = %d: %.2f", T, self.acc_r)
