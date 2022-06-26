"""
Reward calculations.
"""
from ogasched.utils.logging import logging
from ogasched.parameter import param, L, R, K, T
from ogasched import utility
import numpy as np


def element_computation_gain(y_lrk, alpha, utility_func=utility.linear_u):
    """
    Return the computation gain obtained by the device k of the node r for type-l job: f_r^k ( y_(l,r)^k (t) ).
    """
    return utility_func(alpha=alpha, y=y_lrk)


def computation_gain_on_device_k_of_job_type_l(y_lk, nodes_connected, alphas,
                                               utilities=["linear"] * R, d=False):
    """
    Return the computation gain obtained by the device k over all the nodes for type l job if not using derivative.
    Otherwise, return a list of length R, each element is f'_rk(y_lrk).
    :param y_lk: of size R: [y_lk(r=1), ..., y_lk(r=R)]
    :param nodes_connected: a list of node indexes that connected to this job type
    :param alphas: of size R: [r=1, ..., r=R]
    :param utilities: the utility func names of each node for the type-k device
    :param d: use derivative func or not
    :return: a float scalar indicating the gain if d is False. Otherwise, return a numpy array of size R indicating derivatives
    """
    supported_utility_names = param.supported_computation_gain_types
    supported_utilities = param.registered_gains
    supported_utility_derivatives = param.registered_gains_derivative

    if not d:
        res = 0.
        for r_idx in nodes_connected:
            utility_func_name = utilities[r_idx]
            if utility_func_name in supported_utility_names:
                res += element_computation_gain(
                    y_lk[r_idx],
                    alpha=alphas[r_idx],
                    utility_func=supported_utilities[utility_func_name]
                )
            else:
                logging.error("Utility func not supported")
                exit(1)
        return res
    else:
        res = np.zeros(R)
        for r_idx in range(R):
            if r_idx in nodes_connected:
                utility_func_name = utilities[r_idx]
                if utility_func_name in supported_utility_names:
                    res[r_idx] = element_computation_gain(
                        y_lk[r_idx],
                        alpha=alphas[r_idx],
                        utility_func=supported_utility_derivatives[utility_func_name]
                    )
                else:
                    logging.error("Derivative of utility func not supported")
                    exit(1)
        return res


def computation_gain_of_job_type_l(y_l, nodes_connected, alphas=param.alphas,
                                   utilities=[["linear"] * R] * K):
    """
    Return the computation gain for type-l job.
    :param y_l: a 2-dim numpy array of size K * R:
        [[y_l(k=1,r=1), y_l(k=1,r=2), ..., y_l(k=1,r=R)],
         [y_l(k=2,r=1), y_l(k=2,r=2), ..., y_l(k=2,r=R)],
         ...,
         [y_l(k=K,r=1), y_l(k=K,r=2), ..., y_l(k=K,r=R)]]
    :param nodes_connected: a list of node indexes that connected to this job type
    :param alphas: a 2-dim numpy array of size K * R, has the same shape with y_l
    :param utilities: a 2-dim list of size K * R, has the same shape with y_l
    :return: a float scalar indicating the gain
    """
    res = 0.
    for k in range(K):
        res += computation_gain_on_device_k_of_job_type_l(y_l[k, :], nodes_connected,
                                                          alphas=alphas[k, :], utilities=utilities[k])
    return res


def communication_overhead_of_job_type_l(y_l):
    """
    Return the communication overhead for type l job.
    :param y_l: a 2-dim numpy array of size K * R:
        [[y_l(k=1,r=1), y_l(k=1,r=2), ..., y_l(k=1,r=R)],
         [y_l(k=2,r=1), y_l(k=2,r=2), ..., y_l(k=2,r=R)],
         ...,
         [y_l(k=K,r=1), y_l(k=K,r=2), ..., y_l(k=K,r=R)]]
    :return: the max overhead and the corresponding device idx
    """
    betas = param.betas
    max_overhead = -1
    max_k = -1
    for k in range(K):
        overhead = betas[k] * np.sum(y_l[k, :])
        if overhead > max_overhead:
            max_overhead = overhead
            max_k = k
    return max_overhead, max_k


def reward_of_job_type_l(x_l, y_l, nodes_connected, alphas=param.alphas,
                         utilities=[["linear"] * R] * K):
    """
    Calculate the reward of type l job with y_l.
    :param x_l: 0 or 1 to indicate whether the port has a job
    :param y_l: a 2-dim numpy array of size K * R:
        [[y_l(k=1,r=1), y_l(k=1,r=2), ..., y_l(k=1,r=R)],
         [y_l(k=2,r=1), y_l(k=2,r=2), ..., y_l(k=2,r=R)],
         ...,
         [y_l(k=K,r=1), y_l(k=K,r=2), ..., y_l(k=K,r=R)]]
    :param nodes_connected: a list of node indexes that connected to this job type
    :param alphas: a 2-dim numpy array of size K * R, has the same shape with y_l
    :param utilities: a 2-dim list of size K * R, has the same shape with y_l
    :return:
    """
    r = (computation_gain_of_job_type_l(y_l, nodes_connected, alphas, utilities) -
         communication_overhead_of_job_type_l(y_l)[0]) if x_l == 1 else 0
    # TODO: If r is a negative, we need to adjust our parameter settings
    # assert r >= 0
    return r


def reward(x, y, all_nodes_connected, alphas=param.alphas,
           utilities=[["linear"] * R] * K):
    """
    Calculate the reward of y at time t.
    :param x: 0 or 1, of size L: [l=1, ..., l=L]
    :param y: a 3-dim numpy array of size L * (K * R):
        [
            l=1:
            [[y_1(k=1,r=1), y_1(k=1,r=2), ..., y_1(k=1,r=R)],
             [y_1(k=2,r=1), y_1(k=2,r=2), ..., y_1(k=2,r=R)],
             ...,
             [y_1(k=K,r=1), y_1(k=K,r=2), ..., y_1(k=K,r=R)]],

            ...,

            l=L:
            [[y_L(k=1,r=1), y_L(k=1,r=2), ..., y_L(k=1,r=R)],
             [y_L(k=2,r=1), y_L(k=2,r=2), ..., y_L(k=2,r=R)],
             ...,
             [y_L(k=K,r=1), y_L(k=K,r=2), ..., y_L(k=K,r=R)]],
        ]
    :param all_nodes_connected: a 2-dim list of node indexes that connected to each job type
    :param alphas: a 2-dim numpy array of size K * R
    :param utilities: a 2-dim list of size K * R
    :return:
    """
    res = 0.
    for l_idx in range(L):
        res += reward_of_job_type_l(x[l_idx], y[l_idx, :, :], all_nodes_connected[l_idx], alphas, utilities)
    return res


def derivative_of_reward(x, y, all_nodes_connected, alphas=param.alphas,
                         utilities=[["linear"] * R] * K):
    """
    Calculate the derivative of the reward on each element of y, i.e., y_(l,r)^k.
    The returned array d has the same size with y (L * K * R).
    """
    betas = param.betas
    res = np.zeros((L, K, R))

    for l_idx in range(L):
        y_l = y[l_idx, :, :]
        nodes_connected = all_nodes_connected[l_idx]
        if x[l_idx] == 1:
            _, max_k = communication_overhead_of_job_type_l(y_l)
            for k_idx in range(K):
                tmp = computation_gain_on_device_k_of_job_type_l(
                    y_l[k_idx, :],
                    nodes_connected,
                    alphas=alphas[k_idx, :],
                    utilities=utilities[k_idx],
                    d=True
                )
                if k_idx == max_k:
                    res[l_idx, k_idx, :] = tmp - betas[k_idx]
                else:
                    res[l_idx, k_idx, :] = tmp
    return res


def acc_reward(X, Y, all_nodes_connected, alphas=param.alphas,
               utilities=[["linear"] * R] * K):
    """
    Calculate the accumulative reward.
    :param X: a 2-dim numpy array of size T * L:
        [[x_(t=1,l=1), ..., x_(t=1,l=L)],
         ...,
         [x_(t=T,l=1), ..., x_(t=T,l=L)]]
    :param Y: a list of length T, each element is a 3-dim numpy array of size L * (K * R)
    :param all_nodes_connected:
    :param alphas:
    :param utilities:
    :return:
    """
    res = 0.
    for t in range(T):
        res += reward(X[t, :], Y[t], all_nodes_connected, alphas, utilities)
    return res
