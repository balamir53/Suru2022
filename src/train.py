import argparse

from atexit import register
import ray
import random
import time
import math
from fractions import Fraction

#from ray import air, tune
from ray import tune
from ray.tune.registry import register_env

from agents.RiskyValley import RiskyValley
from agents.GolKenari import GolKenari

# from ray.rllib.algorithms.ppo import PPO
import ray.rllib.agents.ppo as ppo

parser = argparse.ArgumentParser(description='Cadet Agents')
parser.add_argument('map', metavar='map', type=str,
                    help='Select Map to Train')
parser.add_argument('--mode', metavar='mode', type=str, default="Train",
                    help='Select Mode[Train,Sim]')
parser.add_argument('--agentBlue', metavar='agentBlue', type=str,
                    help='Class name of Blue Agent')
parser.add_argument('agentRed', metavar='agentRed', type=str,
                    help='Class name of Red Agent')
parser.add_argument('--numOfMatch', metavar='numOfMatch', type=int, nargs='?', default=1,
                    help='Number of matches to play between agents')
parser.add_argument('--render', action='store_true',
                    help='Render the game')
parser.add_argument('--gif', action='store_true',
                    help='Create a gif of the game, also sets render')
parser.add_argument('--img', action='store_true',
                    help='Save images of each turn, also sets render')

args = parser.parse_args()
agents = [None, args.agentRed]

def main():
    #multiple agents multiple policies
    #since there are different kind of agents they have to learn different policies
    #because they have different observation and action spaces (inspect this)
    # ray.init(num_gpus=1, log_to_driver=True, local_mode=True)
    ray.init(local_mode=True)
    register_env("ray", lambda config: GolKenari(args,agents))

    #misconfiguration
    #check the documentation
    config= {"use_critic": True,
                #"log_level": "WARN",
                "num_workers": 0,
                "use_gae": True,
                "lambda": 1.0,
                "kl_coeff": 0.2,
                "rollout_fragment_length": 200,
                "train_batch_size": 4000,
                "sgd_minibatch_size": 128,
                "shuffle_sequences": True,
                "num_sgd_iter": 30,
                "lr": 5e-5,
                "lr_schedule": None,
                "vf_loss_coeff": 1.0,
                "framework": "torch",
                "entropy_coeff": 0.0,
                "entropy_coeff_schedule": None,
                "clip_param": 0.3,
                "vf_clip_param": 10.0,
                "grad_clip": None,
                "kl_target": 0.01,
                "batch_mode": "truncate_episodes",
                "observation_filter": "NoFilter"}

    # Create our RLlib Trainer.
    algo = ppo.PPOTrainer(config=config, env="ray")
    # algo = ppo.PPOTrainer(config=config, env="CartPole-v0")
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # this line works but the saved data doesnt match with current one
    # it finally worked on desktop
    #algo.restore(checkpoint_path="./models/checkpoint_000100/checkpoint-200")

    for _ in range(3):
        print(algo.train())

    # ValueError: Cannot evaluate w/o an evaluation worker set in the Trainer or w/o an env on the local worker!
    # Try one of the following:
    # 1) Set `evaluation_interval` >= 0 to force creating a separate evaluation worker set.
    # 2) Set `create_env_on_driver=True` to force the local (non-eval) worker to have an environment to evaluate on.
    # algo.evaluate()

if __name__ == "__main__":
    main()