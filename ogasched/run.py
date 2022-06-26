"""
The entry.
"""
import psutil
from ogasched.env import Env
from ogasched.utils.logging import logging
from ogasched.parameter import R, K
from ogasched.oga.oga import OGASched
from ogasched.baselines.fairness import Fairness
from ogasched.baselines.binpacking import BinPacking
from ogasched.baselines.spreading import Spreading
from ogasched.baselines.drf import DRF


# Get simulation environment
logging.info("Simulation Environment: \\")
logging.info("    vCPUs: %s" % psutil.cpu_count(logical=True))
logging.info("    %s" % str(psutil.virtual_memory()))


# Run each algorithm
def test_default_env():
    env = Env(
        simplified=True,
        right_d_regular=True,
        utilities=[["linear"] * K] * R
    )
    logging.info("Env instance is created")
    logging.info(str(env))
    # for node in env.nodes:
    #     logging.info(str(node))
    # for job_type in env.job_types:
    #     logging.info(str(job_type))

    oga = OGASched(env)
    bp = BinPacking(env)
    fair = Fairness(env)
    spread = Spreading(env)
    drf = DRF(env)

    oga.run()
    bp.run()
    fair.run()
    spread.run()
    drf.run()


def test_sophisticated_env():
    env = Env(
        simplified=True,
        right_d_regular=True,
        utilities=[["linear"] * K] * R
    )
    logging.info("Env instance is created")
    logging.info(str(env))
    # for node in env.nodes:
    #     logging.info(str(node))
    # for job_type in env.job_types:
    #     logging.info(str(job_type))

    oga = OGASched(env)
    bp = BinPacking(env)
    fair = Fairness(env)
    spread = Spreading(env)
    drf = DRF(env)

    oga.run()
    bp.run()
    fair.run()
    spread.run()
    drf.run()


def test_random_utilities_env():
    env = Env(
        simplified=True,
        right_d_regular=True,
        # TODO: set random utilities
    )
    logging.info("Env instance is created")
    logging.info(str(env))
    # for node in env.nodes:
    #     logging.info(str(node))
    # for job_type in env.job_types:
    #     logging.info(str(job_type))

    oga = OGASched(env)
    bp = BinPacking(env)
    fair = Fairness(env)
    spread = Spreading(env)
    drf = DRF(env)

    oga.run()
    bp.run()
    fair.run()
    spread.run()
    drf.run()


if __name__ == '__main__':
    # test whatever you want
    test_default_env()
