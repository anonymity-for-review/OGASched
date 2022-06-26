"""
Implement the OGASched Algorithm.
"""
import numpy as np
import multiprocessing
from ogasched.parameter import param, L, R, K, T
from ogasched.oga.solution import Y, check_feasible
from ogasched import reward
from ogasched.utils.logging import logging
from ogasched.utils.utils import get_run_time


# TODO: the element in y that no job yields should be filtered out directly! Check that.


class OGASched(object):
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

        # TODO: arrange the nodes with ascending order of utilities
        self.all_nodes_connected = [job_type.nodes_connected for job_type in self.env.job_types]  # of length L
        self.all_jobs_connected = [node.jobs_connected for node in self.env.nodes]  # of length R

        self.all_utilities = [["linear"] * R] * K  # of size K * R
        for r_idx in range(len(self.env.nodes)):
            node = self.env.nodes[r_idx]
            for k_idx in range(K):
                self.all_utilities[k_idx][r_idx] = node.utilities[k_idx]

        # NOTE: define them as properties for parallel running
        # job arrivals of size T * L
        self.xs = self.env.X
        # solutions at each time t
        self.ys = np.zeros((T, L, K, R))
        self.y_hats = np.zeros((T, L, K, R))
        # rewards at each time t
        self.rs = np.zeros(T)
        self.acc_r = 0.
        # the repeat loop's execution times
        self.all_repeat_times = np.ones((T, R, K))

    @get_run_time
    def run(self):
        """
        Run the OGASched algorithm.
        :return:
        """
        t = 0

        # init y and its reward
        solution_y = Y()
        solution_y.init_value(self.xs[t], self.A, self.C, self.all_jobs_connected)
        y = solution_y.values
        self.ys[t] = y

        r = reward.reward(self.xs[t], y, self.all_nodes_connected, utilities=self.all_utilities)
        self.acc_r += r
        self.rs[t] = r

        # init eta and decay rate
        eta = param.init_eta
        decay_rate = param.decay_rate

        for t in range(1, T):
            x = self.xs[t]
            dy = reward.derivative_of_reward(x, y, self.all_nodes_connected, utilities=self.all_utilities)
            z = y + eta * dy

            for r_idx in range(R):
                for k_idx in range(K):
                    # NOTE here is a deep copy. The change on z_rk will not affect z.
                    z_rk = z[:, k_idx, r_idx]
                    # sort all the job types
                    # TODO: Here we can speed up. The job types whose y is zero will always be arranged at the last place.
                    sorted_z_rk = sorted(z_rk, reverse=True)
                    sorted_l_indexes = np.argsort(-z_rk)

                    B_rk_1 = set()
                    B_rk_2 = set()
                    B_rk_3 = set(list(range(L)))

                    while True:
                        # calculate rho_r^k
                        tmp1 = np.sum([sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] for l_idx in B_rk_3])
                        tmp2 = np.sum([self.A[l_idx][k_idx] for l_idx in B_rk_1])
                        half_rho_rk = (tmp1 - self.C[r_idx][k_idx] + tmp2) / len(B_rk_3)

                        S_rk = set()
                        # update y_hat and S_rk
                        for l_idx in sorted_l_indexes:
                            if l_idx in B_rk_1:
                                self.y_hats[t][l_idx][k_idx][r_idx] = self.A[l_idx][k_idx]
                            elif l_idx in B_rk_3:
                                self.y_hats[t][l_idx][k_idx][r_idx] = \
                                    sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] - half_rho_rk
                                if self.y_hats[t][l_idx][k_idx][r_idx] < 0:
                                    # S_rk should be a subset of sorted_l_indexes since l_idx
                                    S_rk = set([
                                        sorted_l_indexes[i] for i in
                                        range(np.argwhere(sorted_l_indexes == l_idx)[0][0], len(sorted_l_indexes))
                                    ])
                                    break

                        # Update three sets
                        B_rk_3.difference_update(S_rk)
                        B_rk_2 = B_rk_2.union(S_rk)

                        if not bool(S_rk):
                            break

                    while True:
                        first_l = sorted_l_indexes[0]
                        if self.y_hats[t][first_l][k_idx][r_idx] <= self.A[first_l][k_idx]:
                            break

                        self.all_repeat_times[t, r_idx, k_idx] += 1

                        B_rk_1 = set([first_l])
                        B_rk_3 = set(list(range(L)))
                        B_rk_3.difference_update([first_l])

                        # TODO: Just copied from the above. Use a more elegant implementation!
                        while True:
                            # calculate rho_r^k
                            tmp1 = sum([sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] for l_idx in B_rk_3])
                            tmp2 = sum([self.A[l_idx][k_idx] for l_idx in B_rk_1])
                            half_rho_rk = (tmp1 - self.C[r_idx][k_idx] + tmp2) / len(B_rk_3)

                            S_rk = set()
                            # update y_hat and S_rk
                            for l_idx in sorted_l_indexes:
                                if l_idx in B_rk_1:
                                    self.y_hats[t][l_idx][k_idx][r_idx] = self.A[l_idx][k_idx]
                                elif l_idx in B_rk_3:
                                    self.y_hats[t][l_idx][k_idx][r_idx] = \
                                        sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] - half_rho_rk
                                    if self.y_hats[t][l_idx][k_idx][r_idx] < 0:
                                        # S_rk should be a subset of sorted_l_indexes since l_idx
                                        S_rk = set([
                                            sorted_l_indexes[i] for i in
                                            range(np.argwhere(sorted_l_indexes == l_idx)[0][0], len(sorted_l_indexes))
                                        ])
                                        break

                            # Update three sets
                            B_rk_3.difference_update(S_rk)
                            B_rk_2 = B_rk_2.union(S_rk)

                            if not bool(S_rk):
                                break

            # TODO: this may not right!
            for l_idx in range(L):
                if x[l_idx] == 0:
                    self.y_hats[t][l_idx] = 0.

            # use y_hat to update y
            # check_feasible(self.y_hats[t], x, self.A, self.C, self.all_jobs_connected)
            y = self.y_hats[t]

            self.ys[t] = y
            self.acc_r += r
            self.rs[t] = reward.reward(x, y, self.all_nodes_connected, utilities=self.all_utilities)

            logging.info("[OGASched] Reward at t = %d: %.2f", t, self.rs[t])

            eta *= decay_rate

        logging.info("[OGASched] Acc. Reward within T = %d: %.2f", T, self.acc_r)

    @get_run_time
    def run_in_parallel(self):
        """
        Run OGASched with R * K async sub-processes.
        :return:
        """
        t = 0

        # init y and its reward
        solution_y = Y()
        solution_y.init_value(self.xs[t], self.A, self.C, self.all_jobs_connected)
        y = solution_y.values
        self.ys[t] = y

        r = reward.reward(self.xs[t], y, self.all_nodes_connected, utilities=self.all_utilities)
        self.acc_r += r
        self.rs[t] = r

        # init eta and decay rate
        eta = param.init_eta
        decay_rate = param.decay_rate

        for t in range(1, T):
            x = self.xs[t]
            dy = reward.derivative_of_reward(x, y, self.all_nodes_connected, utilities=self.all_utilities)
            z = y + eta * dy

            # run async
            p = multiprocessing.Pool(R * K)
            for r_idx in range(R):
                for k_idx in range(K):
                    p.apply_async(self.worker, args=(t, r_idx, k_idx, z[:, k_idx, r_idx]))
            p.close()
            p.join()

            # TODO: this may not right!
            for l_idx in range(L):
                if x[l_idx] == 0:
                    self.y_hats[t][l_idx] = 0.

            # use y_hat to update y
            check_feasible(self.y_hats[t], x, self.A, self.C, self.all_jobs_connected)
            y = self.y_hats[t]

            self.ys[t] = y
            self.acc_r += r
            self.rs[t] = reward.reward(x, y, self.all_nodes_connected, utilities=self.all_utilities)

            logging.info("[OGASched] Reward at t = %d: %.2f", t, self.rs[t])

            eta *= decay_rate

        logging.info("[OGASched] Acc. Reward within T = %d: %.2f", T, self.acc_r)

    def worker(self, t, r_idx, k_idx, z_rk):
        # sort all the job types
        # TODO: Here we can speed up. The job types whose y is zero will always be arranged at the last place.
        sorted_z_rk = sorted(z_rk, reverse=True)
        sorted_l_indexes = np.argsort(-z_rk)

        B_rk_1 = set()
        B_rk_2 = set()
        B_rk_3 = set(list(range(L)))

        while True:
            # calculate rho_r^k
            tmp1 = sum([sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] for l_idx in B_rk_3])
            tmp2 = sum([self.A[l_idx][k_idx] for l_idx in B_rk_1])
            half_rho_rk = (tmp1 - self.C[r_idx][k_idx] + tmp2) / len(B_rk_3)

            S_rk = set()
            # update y_hat and S_rk
            for l_idx in sorted_l_indexes:
                if l_idx in B_rk_1:
                    self.y_hats[t][l_idx][k_idx][r_idx] = self.A[l_idx][k_idx]
                elif l_idx in B_rk_3:
                    self.y_hats[t][l_idx][k_idx][r_idx] = \
                        sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] - half_rho_rk
                    if self.y_hats[t][l_idx][k_idx][r_idx] < 0:
                        # S_rk should be a subset of sorted_l_indexes since l_idx
                        S_rk = set([
                            sorted_l_indexes[i] for i in
                            range(np.argwhere(sorted_l_indexes == l_idx)[0][0], len(sorted_l_indexes))
                        ])
                        break

            # Update three sets
            B_rk_3.difference_update(S_rk)
            B_rk_2 = B_rk_2.union(S_rk)

            if not bool(S_rk):
                break

        while True:
            first_l = sorted_l_indexes[0]
            if self.y_hats[t][first_l][k_idx][r_idx] <= self.A[first_l][k_idx]:
                break

            self.all_repeat_times[t, r_idx, k_idx] += 1

            B_rk_1 = set(first_l)
            B_rk_3 = set(list(range(L)))
            B_rk_3.difference_update(first_l)

            # TODO: Just copied from the above. Use a more elegant implementation!
            while True:
                # calculate rho_r^k
                tmp1 = sum([sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] for l_idx in B_rk_3])
                tmp2 = sum([self.A[l_idx][k_idx] for l_idx in B_rk_1])
                half_rho_rk = (tmp1 - self.C[r_idx][k_idx] + tmp2) / len(B_rk_3)

                S_rk = set()
                # update y_hat and S_rk
                for l_idx in sorted_l_indexes:
                    if l_idx in B_rk_1:
                        self.y_hats[t][l_idx][k_idx][r_idx] = self.A[l_idx][k_idx]
                    elif l_idx in B_rk_3:
                        self.y_hats[t][l_idx][k_idx][r_idx] = \
                            sorted_z_rk[np.argwhere(sorted_l_indexes == l_idx)[0][0]] - half_rho_rk
                        if self.y_hats[t][l_idx][k_idx][r_idx] < 0:
                            # S_rk should be a subset of sorted_l_indexes since l_idx
                            S_rk = set([
                                sorted_l_indexes[i] for i in
                                range(np.argwhere(sorted_l_indexes == l_idx)[0][0], len(sorted_l_indexes))
                            ])
                            break

                # Update three sets
                B_rk_3.difference_update(S_rk)
                B_rk_2 = B_rk_2.union(S_rk)

                if not bool(S_rk):
                    break

    def regret_upper_bound(self):
        """
        Return the upper bound of the regret of the OGA policy.
        :return:
        """
        a_bar = np.max(self.A, axis=0)
        max_beta = np.max(param.betas)
        varpi_ls = []
        for r_idx in range(len(self.env.nodes)):
            node = self.env.nodes[r_idx]
            varpi_ls.append([param.get_varpi(node.utilities[k_idx], param.alphas[k_idx, r_idx]) for k_idx in range(K)])
        varpi_ls = np.array(varpi_ls)
        max_varpi_ls = np.max(varpi_ls, axis=1)

        tmp1 = 0.
        for k_idx in range(K):
            for r_idx in range(R):
                tmp1 += a_bar[k_idx] * self.env.nodes[r_idx].c[k_idx]
        tmp1 = np.sqrt(tmp1 * 2 * T)

        tmp2 = 0.
        max_beta_squared = max_beta ** 2
        for l_idx in range(L):
            for r_idx in self.env.job_types[l_idx].nodes_connected:
                tmp2 += max_beta_squared + K * max_varpi_ls[r_idx] ** 2
        tmp2 = np.sqrt(tmp2)

        return tmp1 * tmp2
