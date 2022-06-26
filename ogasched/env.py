"""
Construction of the experiment environment.
"""
import numpy as np
import random
from ogasched.parameter import param, L, R, K, T
from ogasched.utils import utils


class Node(object):
    def __init__(self, vcpucore_num, mem_size, gpu_num=0, tpu_num=0, npu_num=0, fpga_num=0,
                 utilities=["linear"] * K):
        self.vcpucore_num = vcpucore_num
        self.mem_size = mem_size
        self.gpu_num = gpu_num
        self.tpu_num = tpu_num
        self.npu_num = npu_num
        self.fpga_num = fpga_num
        self.equipped_res_types = 2

        # jobs_connected stores the job type indexes that connect to this node
        self.jobs_connected = []
        # c is a numpy array of size K
        self.c = np.array([
            self.vcpucore_num, self.mem_size, self.gpu_num,
            self.tpu_num, self.npu_num, self.fpga_num
        ])
        self.utilities = utilities

    def update_equipped_res_types(self):
        """
        Update the number of types of computation devices this node has.
        """
        cnt = 2
        if self.gpu_num > 0:
            cnt += 1
        if self.tpu_num > 0:
            cnt += 1
        if self.npu_num > 0:
            cnt += 1
        if self.fpga_num > 0:
            cnt += 1
        self.equipped_res_types = cnt

    def __str__(self):
        return "Node info: \\\n    %d equipped devices\n    number of each: %s\n    utility of each: %s\n" \
               "    Job type indexes connected: %s\n" % (
                   self.equipped_res_types, self.c, self.utilities, self.jobs_connected)


class JobType(object):
    def __init__(self, req_vcpucore_num, req_mem_size, req_gpu_num, req_tpu_num=0, req_npu_num=0, req_fpga_num=0):
        self.req_vcpucore_num = req_vcpucore_num
        self.req_mem_size = req_mem_size
        self.req_gpu_num = req_gpu_num
        self.req_tpu_num = req_tpu_num
        self.req_npu_num = req_npu_num
        self.req_fpga_num = req_fpga_num

        # nodes_connected stores the node indexes that connect to this job type
        self.nodes_connected = []
        # a is a numpy array of size K
        self.a = np.array([
            self.req_vcpucore_num, self.req_mem_size, self.req_gpu_num,
            self.req_tpu_num, self.req_npu_num, self.req_fpga_num
        ])

    def __str__(self):
        return "Job type info: \\\n    request on each device: %s\n" \
               "    Node indexes connected: %s\n" % (self.a, self.nodes_connected)


class BipartiteGraph(object):
    def __init__(self, simplified=True):
        """
        Initialization.
        :param simplified: If simplified, each node will only be equipped with at most 3-dim resources,
            and each job request will only request 3-dim resources (CPU, MEM, GPU).
        """
        self.simplified = simplified
        self.nodes = []  # a list of Node instances
        self.job_types = []  # a list of JobType instances
        self.channels = []  # a list of 2-value tuple: (job_type_idx, node_idx)
        self.Lrs = []  # a list of the length `len(self.nodes)` (for each node)
        self.Rls = []  # a list of the length `len(self.job_types)` (for each port)

    def get_channel_num(self):
        """
        Return the complexity of the graph.
        :return:
        """
        return len(self.channels)

    def generate_graph_from_trace(self):
        pass

    def generate_graph(self, right_d_regular=True,
                       utilities=[["linear"] * K] * R):
        """
        This function will set the bipartite graph for online scheduling.
        Specifically, it does the following things:
            (1) generate computing instances (nodes);
            (2) generate job types;
            (3) set channels with generated service locality constraint
        :param right_d_regular:
        :param utilities:
        :return:
        """
        # step 1: generate nodes
        vcpucore_nums = utils.sample_from_uniform_int(
            param.min_vcpucore_num,
            param.max_vcpucore_num,
            R
        )
        mem_sizes = utils.sample_from_uniform_int(
            param.min_mem_size,
            param.max_mem_size,
            R
        )
        gpu_nums = utils.sample_from_uniform_int(
            param.min_gpu_num,
            param.max_gpu_num,
            R
        )
        tpu_nums = utils.sample_from_uniform_int(
            param.min_tpu_num,
            param.max_tpu_num,
            R
        )
        npu_nums = utils.sample_from_uniform_int(
            param.min_npu_num,
            param.max_npu_num,
            R
        )
        fpga_nums = utils.sample_from_uniform_int(
            param.min_fpga_num,
            param.max_fpga_num,
            R
        )

        for r_idx in range(R):
            if self.simplified:
                node = Node(
                    vcpucore_num=vcpucore_nums[r_idx],
                    mem_size=mem_sizes[r_idx],
                    gpu_num=gpu_nums[r_idx],
                    utilities=utilities[r_idx]
                )
            else:
                node = Node(
                    vcpucore_num=vcpucore_nums[r_idx],
                    mem_size=mem_sizes[r_idx],
                    gpu_num=gpu_nums[r_idx],
                    tpu_num=tpu_nums[r_idx],
                    npu_num=npu_nums[r_idx],
                    fpga_num=fpga_nums[r_idx],
                    utilities=utilities[r_idx]
                )
            node.update_equipped_res_types()
            self.nodes.append(node)

        # step 2: generate job types
        contention_level = param.contention_level
        req_vcpucore_nums = contention_level * utils.sample_from_uniform(
            param.min_vcpucore_num,
            param.max_vcpucore_num,
            L
        )
        req_mem_sizes = contention_level * utils.sample_from_uniform(
            param.min_mem_size,
            param.max_mem_size,
            L
        )
        req_gpu_nums = contention_level * utils.sample_from_uniform(
            param.min_gpu_num,
            param.max_gpu_num,
            L
        )
        req_tpu_nums = contention_level * utils.sample_from_uniform(
            param.min_tpu_num,
            param.max_tpu_num,
            L
        )
        req_npu_nums = contention_level * utils.sample_from_uniform(
            param.min_npu_num,
            param.max_npu_num,
            L
        )
        req_fpga_nums = contention_level * utils.sample_from_uniform(
            param.min_fpga_num,
            param.max_fpga_num,
            L
        )

        for l_idx in range(L):
            if self.simplified:
                job_type = JobType(
                    req_vcpucore_num=req_vcpucore_nums[l_idx],
                    req_mem_size=req_mem_sizes[l_idx],
                    req_gpu_num=req_gpu_nums[l_idx],
                )
            else:
                job_type = JobType(
                    req_vcpucore_num=req_vcpucore_nums[l_idx],
                    req_mem_size=req_mem_sizes[l_idx],
                    req_gpu_num=req_gpu_nums[l_idx],
                    req_tpu_num=req_tpu_nums[l_idx],
                    req_npu_num=req_npu_nums[l_idx],
                    req_fpga_num=req_fpga_nums[l_idx]
                )
            self.job_types.append(job_type)

        # step 3: set service locality
        if right_d_regular:
            self.set_right_d_regular_channels()
        else:
            self.set_random_channels()
        self.set_Lr()
        self.set_Rl()

    def set_random_channels(self):
        connected_job_types_num = utils.sample_from_uniform_int(
            param.min_connected_job_types,
            param.max_connected_job_types,
            R
        )
        # set jobs_connected of each node, nodes_connected of each job, and channels
        for r_idx in range(R):
            node = self.nodes[r_idx]
            # randomly sample a specific number of job types that connect to this node
            node.jobs_connected = random.sample(range(L), connected_job_types_num[r_idx])
            for l_idx in node.jobs_connected:
                self.channels.append((l_idx, r_idx))
                job_type = self.job_types[l_idx]
                job_type.nodes_connected.append(r_idx)

    def set_right_d_regular_channels(self):
        connected_job_types_num = utils.sample_from_uniform_int(
            param.min_connected_job_types,
            param.max_connected_job_types,
            1
        )
        # set jobs_connected of each node, nodes_connected of each job, and channels
        for r_idx in range(R):
            node = self.nodes[r_idx]
            # randomly sample a specific number of job types that connect to this node
            node.jobs_connected = random.sample(range(L), connected_job_types_num[0])
            for l_idx in node.jobs_connected:
                self.channels.append((l_idx, r_idx))
                job_type = self.job_types[l_idx]
                job_type.nodes_connected.append(r_idx)

    def set_Lr(self):
        for node in self.nodes:
            self.Lrs.append(node.jobs_connected)

    def set_Rl(self):
        for job_type in self.job_types:
            self.Rls.append(job_type.nodes_connected)

    def __str__(self):
        return "    %d nodes, %d job types\n    Is a simplified scenario? %s\n" % (
            len(self.nodes), len(self.job_types), "True" if self.simplified else "False")


class Env(BipartiteGraph):
    def __init__(self, simplified=True, right_d_regular=True,
                 utilities=[["linear"] * K] * R):
        super().__init__(simplified)
        self.right_d_regular = right_d_regular

        self.X = np.zeros((T, L))
        self.generate_job_arrivals()

        self.generate_graph(right_d_regular, utilities)

    def generate_job_arrivals(self):
        for t in range(T):
            self.X[t, :] = utils.sample_from_bernoulli(L, param.job_arrival_prob)

    def __str__(self):
        s = "Env info: \\\n"
        s += super().__str__()
        s += "    Is right-d regular? %s\n" % self.right_d_regular
        return s
