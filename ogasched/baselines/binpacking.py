"""
The binpacking algorithm.
A node with higher resource utilization will have a higher score. So that resource fragmentation could be avoided.
BinPacking and Spreading are optional scheduling plugins in K8S.
"""
import numpy as np
from ogasched.parameter import L, R, K, T
from ogasched.oga.solution import check_feasible
from ogasched import reward
from ogasched.utils.logging import logging
from ogasched.utils.utils import get_run_time


class BinPacking(object):
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
        Binpacking dispatches job requests to the node who can handle the requests with the minimum idle resources.
        :return:
        """
        for t in range(0, T):
            # remained_res is of size R * K
            remained_res = [node.c for node in self.env.nodes]
            x = self.xs[t]

            for l_idx in range(L):
                if x[l_idx] == 0:
                    continue
                for k_idx in range(K):
                    nodes_res_left_sorted = np.sort(remained_res)
                    nodes_indexes_sorted = np.argsort(remained_res)

                    request = self.env.job_types[l_idx].a[k_idx]
                    # find the node that can satisfy the request but with minimum allocatable
                    chosen_r = -1
                    for idle in nodes_res_left_sorted[:, k_idx]:
                        if idle >= request:
                            chosen_r += 1
                            break
                    if chosen_r == -1:
                        # no node satisfy the request
                        # in this case, chose the node with the largest left resources,
                        # and allocate all of them to this request
                        chosen_r = nodes_indexes_sorted[-1, k_idx]
                        self.ys[t][l_idx][k_idx][chosen_r] = remained_res[chosen_r][k_idx]
                        remained_res[chosen_r][k_idx] = 0
                    elif chosen_r == len(nodes_indexes_sorted[:, k_idx]) - 1:
                        if nodes_res_left_sorted[-1, k_idx] < request:
                            # no node satisfy the request
                            # in this case, chose the node with the largest left resources
                            chosen_r = nodes_indexes_sorted[-1, k_idx]
                            self.ys[t][l_idx][k_idx][chosen_r] = remained_res[chosen_r][k_idx]
                            remained_res[chosen_r][k_idx] = 0
                        else:
                            # allocate from chosen_r
                            self.ys[t][l_idx][k_idx][chosen_r] = request
                            remained_res[chosen_r][k_idx] -= request
                    else:
                        # allocate from chosen_r
                        self.ys[t][l_idx][k_idx][chosen_r] = request
                        remained_res[chosen_r][k_idx] -= request

            # check_feasible(self.ys[t], x, self.A, self.C, self.all_jobs_connected)
            re = reward.reward(x, self.ys[t], self.all_nodes_connected, utilities=self.all_utilities)
            self.rs[t] = re
            self.acc_r += re
            logging.info("[BinPacking] Reward at t = %d: %.2f", t, self.rs[t])

        logging.info("[BinPacking] Acc. Reward within T = %d: %.2f", T, self.acc_r)
