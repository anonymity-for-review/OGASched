"""
Parameter settings for constructing the Bipartite Scheduling environment.
"""
import types
import numpy as np
from ogasched.utils import utils
from ogasched import utility
from ogasched.utils.logging import logging


class Parameter(object):
    def __init__(self, time_slot_num=10 + 1,
                 job_type_num=20, node_num=20,
                 min_vcpucore_num=1, max_vcpucore_num=128,
                 min_mem_size=2, max_mem_size=512,
                 min_gpu_num=0, max_gpu_num=64,
                 min_tpu_num=0, max_tpu_num=8,
                 min_npu_num=0, max_npu_num=8,
                 min_fpga_num=0, max_fpga_num=4,
                 job_arrival_prob=0.8,
                 contention_level=0.4,  # NOTE: set this param carefully
                 min_connected_job_types=1, max_connected_job_types=10,
                 supported_computation_gain_types=None,
                 min_alpha=1, max_alpha=1.5,
                 min_beta=0.01, max_beta=0.1,
                 init_eta=0.999, decay_rate=0.99):
        """
        Parameter initialization.
        """
        self.__time_slot_num = time_slot_num
        self.__job_type_num = job_type_num
        self.__node_num = node_num

        # for different types of computing resources
        self.__min_vcpucore_num = min_vcpucore_num
        self.__max_vcpucore_num = max_vcpucore_num

        # size of Gi
        self.__min_mem_size = min_mem_size
        self.__max_mem_size = max_mem_size

        self.__min_gpu_num = min_gpu_num
        self.__max_gpu_num = max_gpu_num

        self.__min_tpu_num = min_tpu_num
        self.__max_tpu_num = max_tpu_num

        self.__min_npu_num = min_npu_num
        self.__max_npu_num = max_npu_num

        self.__min_fpga_num = min_fpga_num
        self.__max_fpga_num = max_fpga_num

        self.__device_type_num = 6

        self.__job_arrival_prob = job_arrival_prob
        self.__contention_level = contention_level

        self.__min_connected_job_types = min_connected_job_types
        self.__max_connected_job_types = max_connected_job_types

        # default gain can be linear, log, poly, or reciprocal
        #   - linear: y = a * x
        #   - log: y = a * ln(x+1)
        #   - poly: y = a * sqrt(x + 1)
        #   - reciprocal: y = 1/a - 1/(x + a)
        if supported_computation_gain_types is None:
            self.__supported_computation_gain_types = ["linear", "log", "poly", "reciprocal"]
            self.__registered_gains = {
                "linear": utility.linear_u,
                "log": utility.log_u,
                "poly": utility.poly_u,
                "reciprocal": utility.reciprocal_u
            }
            self.__registered_gains_derivative = {
                "linear": utility.derivative_linear_u,
                "log": utility.derivative_log_u,
                "poly": utility.derivative_poly_u,
                "reciprocal": utility.derivative_reciprocal_u
            }

        # alpha is the coefficient in the computation gain
        self.__min_alpha = min_alpha
        self.__max_alpha = max_alpha
        # __alphas is a numpy array of size K * R
        self.__alphas = [0] * (self.__node_num * self.__device_type_num)
        self.generate_alphas()

        # beta is the coefficient of the communication overhead
        self.__min_beta = min_beta
        self.__max_beta = max_beta
        # __betas is a numpy array of size K
        self.__betas = [0] * self.__device_type_num
        self.generate_betas()

        self.__init_eta = init_eta
        self.__decay_rate = decay_rate

    @property
    def time_slot_num(self):
        return self.__time_slot_num

    @time_slot_num.setter
    def time_slot_num(self, time_slot_num):
        if isinstance(time_slot_num, int):
            self.__time_slot_num = time_slot_num
        else:
            logging.error(ValueError("Time slot number should be integer"))

    @property
    def job_type_num(self):
        return self.__job_type_num

    @job_type_num.setter
    def job_type_num(self, job_types_num):
        if isinstance(job_types_num, int):
            self.__job_type_num = job_types_num
        else:
            logging.error(ValueError("Job type number should be integer"))

    @property
    def node_num(self):
        return self.__node_num

    @node_num.setter
    def node_num(self, node_num):
        if isinstance(node_num, int):
            self.__node_num = node_num
        else:
            logging.error(ValueError("Node number should be integer"))

    @property
    def min_vcpucore_num(self):
        return self.__min_vcpucore_num

    @min_vcpucore_num.setter
    def min_vcpucore_num(self, min_vcpucore_num):
        if isinstance(min_vcpucore_num, int):
            self.__min_vcpucore_num = min_vcpucore_num
        else:
            logging.error(ValueError("Min virtual CPU core number should be integer"))

    @property
    def max_vcpucore_num(self):
        return self.__max_vcpucore_num

    @max_vcpucore_num.setter
    def max_vcpucore_num(self, max_vcpucore_num):
        if isinstance(max_vcpucore_num, int):
            self.__max_vcpucore_num = max_vcpucore_num
        else:
            logging.error(ValueError("Max virtual CPU core number should be integer"))

    @property
    def min_mem_size(self):
        return self.__min_mem_size

    @min_mem_size.setter
    def min_mem_size(self, min_mem_size):
        if isinstance(min_mem_size, int):
            self.__min_mem_size = min_mem_size
        else:
            logging.error(ValueError("Min MEM size should be integer"))

    @property
    def max_mem_size(self):
        return self.__max_mem_size

    @max_mem_size.setter
    def max_mem_size(self, max_mem_size):
        if isinstance(max_mem_size, int):
            self.__max_mem_size = max_mem_size
        else:
            logging.error(ValueError("Max MEM size should be integer"))

    @property
    def min_gpu_num(self):
        return self.__min_gpu_num

    @min_gpu_num.setter
    def min_gpu_num(self, min_gpu_num):
        if isinstance(min_gpu_num, int):
            self.__min_gpu_num = min_gpu_num
        else:
            logging.error(ValueError("Min GPU number should be integer"))

    @property
    def max_gpu_num(self):
        return self.__max_gpu_num

    @max_gpu_num.setter
    def max_gpu_num(self, max_gpu_num):
        if isinstance(max_gpu_num, int):
            self.__max_gpu_num = max_gpu_num
        else:
            logging.error(ValueError("Max GPU number should be integer"))

    @property
    def min_tpu_num(self):
        return self.__min_tpu_num

    @min_tpu_num.setter
    def min_tpu_num(self, min_tpu_num):
        if isinstance(min_tpu_num, int):
            self.__min_tpu_num = min_tpu_num
        else:
            logging.error(ValueError("Min TPU number should be integer"))

    @property
    def max_tpu_num(self):
        return self.__max_tpu_num

    @max_tpu_num.setter
    def max_tpu_num(self, max_tpu_num):
        if isinstance(max_tpu_num, int):
            self.__max_tpu_num = max_tpu_num
        else:
            logging.error(ValueError("Max TPU number should be integer"))

    @property
    def min_npu_num(self):
        return self.__min_npu_num

    @min_npu_num.setter
    def min_npu_num(self, min_npu_num):
        if isinstance(min_npu_num, int):
            self.__min_npu_num = min_npu_num
        else:
            logging.error(ValueError("Min NPU number should be integer"))

    @property
    def max_npu_num(self):
        return self.__max_npu_num

    @max_npu_num.setter
    def max_npu_num(self, max_npu_num):
        if isinstance(max_npu_num, int):
            self.__max_npu_num = max_npu_num
        else:
            logging.error(ValueError("Max NPU number should be integer"))

    @property
    def min_fpga_num(self):
        return self.__min_fpga_num

    @min_fpga_num.setter
    def min_fpga_num(self, min_fpga_num):
        if isinstance(min_fpga_num, int):
            self.__min_fpga_num = min_fpga_num
        else:
            logging.error(ValueError("Min FPGA number should be integer"))

    @property
    def max_fpga_num(self):
        return self.__max_fpga_num

    @max_fpga_num.setter
    def max_fpga_num(self, max_fpga_num):
        if isinstance(max_fpga_num, int):
            self.__max_fpga_num = max_fpga_num
        else:
            logging.error(ValueError("Max FPGA number should be integer"))

    @property
    def device_type_num(self):
        return self.__device_type_num

    @property
    def job_arrival_prob(self):
        return self.__job_arrival_prob

    @job_arrival_prob.setter
    def job_arrival_prob(self, job_arrival_prob):
        if isinstance(job_arrival_prob, float) and 0 < job_arrival_prob <= 1.:
            self.__job_arrival_prob = job_arrival_prob
        else:
            logging.error(ValueError("Job arrival probability should be a float in (0, 1]"))

    @property
    def contention_level(self):
        return self.__contention_level

    @contention_level.setter
    def contention_level(self, contention_level):
        if isinstance(contention_level, float) and 0 < contention_level < 1.0:
            self.__contention_level = contention_level
        else:
            logging.error(ValueError("Contention level should be a float in (0, 1)"))

    @property
    def min_connected_job_types(self):
        return self.__min_connected_job_types

    @min_connected_job_types.setter
    def min_connected_job_types(self, min_connected_job_types):
        if isinstance(min_connected_job_types, int):
            self.__min_connected_job_types = min_connected_job_types
        else:
            logging.error(ValueError("Min connected job types should be integer"))

    @property
    def max_connected_job_types(self):
        return self.__max_connected_job_types

    @max_connected_job_types.setter
    def max_connected_job_types(self, max_connected_job_types):
        if isinstance(max_connected_job_types, int):
            self.__max_connected_job_types = max_connected_job_types
        else:
            logging.error(ValueError("Max connected job types should be integer"))

    @property
    def supported_computation_gain_types(self):
        return self.__supported_computation_gain_types

    def add_gain_type(self, gain_type, gain_u, gain_u_derivative):
        """
        Register computation gain.
        :param gain_type: the function name
        :param gain_u: the function signature of the utility
        :param gain_u_derivative: the derivative function signature of the utility
        :return:
        """
        if isinstance(gain_type, str) and isinstance(gain_u, types.FunctionType) and \
                isinstance(gain_u_derivative, types.FunctionType):
            self.__supported_computation_gain_types.append(gain_type)
            self.__registered_gains[gain_type] = gain_u
            self.__registered_gains_derivative[gain_type] = gain_u_derivative
        else:
            logging.error(ValueError("Illegal utility name or func"))

    @property
    def registered_gains(self):
        return self.__registered_gains

    @property
    def registered_gains_derivative(self):
        return self.__registered_gains_derivative

    @property
    def min_alpha(self):
        return self.__min_alpha

    @min_alpha.setter
    def min_alpha(self, min_alpha):
        if isinstance(min_alpha, float):
            self.__min_alpha = min_alpha
        else:
            logging.error(ValueError("Min alpha should be float"))

    @property
    def max_alpha(self):
        return self.__max_alpha

    @max_alpha.setter
    def max_alpha(self, max_alpha):
        if isinstance(max_alpha, float):
            self.__max_alpha = max_alpha
        else:
            logging.error(ValueError("Max alpha should be float"))

    def generate_alphas(self):
        self.__alphas = utils.sample_from_uniform(
            self.__min_alpha,
            self.__max_alpha,
            self.__node_num * self.__device_type_num
        )
        self.__alphas = np.resize(self.__alphas, (self.__device_type_num, self.__node_num))

    @property
    def alphas(self):
        return self.__alphas

    @property
    def min_beta(self):
        return self.__min_beta

    @min_beta.setter
    def min_beta(self, min_beta):
        if isinstance(min_beta, float):
            self.__min_beta = min_beta
        else:
            logging.error(ValueError("Min beta should be float"))

    @property
    def max_beta(self):
        return self.__max_beta

    @max_beta.setter
    def max_beta(self, max_beta):
        if isinstance(max_beta, float):
            self.__max_beta = max_beta
        else:
            logging.error(ValueError("Max beta should be float"))

    def generate_betas(self):
        self.__betas = utils.sample_from_uniform(
            self.__min_beta,
            self.__max_beta,
            self.__device_type_num
        )

    @property
    def betas(self):
        return self.__betas

    @property
    def init_eta(self):
        return self.__init_eta

    @init_eta.setter
    def init_eta(self, init_eta):
        if isinstance(init_eta, float) and 0 < init_eta < 1:
            self.__init_eta = init_eta
        else:
            logging.error(ValueError("Init eta should be a float in (0, 1)"))

    @property
    def decay_rate(self):
        return self.__decay_rate

    @decay_rate.setter
    def decay_rate(self, decay_rate):
        if isinstance(decay_rate, float) and 0 < decay_rate < 1:
            self.__decay_rate = decay_rate
        else:
            logging.error(ValueError("Decay rate should be a float in (0, 1)"))

    def get_varpi(self, utility_name, alpha):
        """
        The upper bound of the derivative of the utility at y = 0.
        :param utility_name:
        :param alpha:
        :return:
        """
        if utility_name == "linear" or utility_name == "log":
            return alpha
        elif utility_name == "poly":
            return 0.5 * alpha
        elif utility_name == "reciprocal":
            return 1 / (alpha ** 2)
        elif utility_name in self.supported_computation_gain_types:
            return self.registered_gains_derivative[utility_name](alpha=alpha, y=0)
        else:
            logging.error(AttributeError("Input utility func not registered"))


# the global Parameter instance
param = Parameter()
logging.info("Global parameter is initialized")

# global variables that used everywhere
L = param.job_type_num
R = param.node_num
K = param.device_type_num
T = param.time_slot_num
EPS = 0.000001
