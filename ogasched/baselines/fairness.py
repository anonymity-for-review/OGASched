"""
Fairness is supported by Volcano in the manner of plugins.
"""
import numpy as np
from ogasched.parameter import L, R, K, T
from ogasched import reward
from ogasched.oga.solution import check_feasible
from ogasched.utils.logging import logging
from ogasched.utils.utils import get_run_time


class Fairness(object):
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
        At each time t, allocate computing devices to jobs based on the ratio of their requests.
        :return:
        """
        for t in range(0, T):
            x = self.xs[t]
            for r_idx in range(R):
                # Lr_of_interested stores the job indexes that yield job and processable to node r
                Lr_of_interested = [l_idx for l_idx in self.all_jobs_connected[r_idx] if x[l_idx] == 1]
                for k_idx in range(K):
                    if self.C[r_idx][k_idx] == 0:
                        # If zero, this node does not have type-k device
                        continue
                    all_requested = sum([self.A[l_idx][k_idx] for l_idx in Lr_of_interested])
                    for l_idx in Lr_of_interested:
                        self.ys[t][l_idx][k_idx][r_idx] = min(
                            self.A[l_idx][k_idx] / all_requested * self.C[r_idx][k_idx],
                            self.A[l_idx][k_idx]
                        )

            # check whether y is feasible and update the reward
            # check_feasible(self.ys[t], x, self.A, self.C, self.all_jobs_connected)
            re = reward.reward(self.xs[t], self.ys[t], self.all_nodes_connected, utilities=self.all_utilities)
            self.rs[t] = re
            self.acc_r += re
            logging.info("[Fairness] Reward at t = %d: %.2f", t, self.rs[t])

        logging.info("[Fairness] Acc. Reward within T = %d: %.2f", T, self.acc_r)
