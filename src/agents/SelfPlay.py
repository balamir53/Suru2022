from random import randint,random
import copy
import numpy as np
from utilities import *
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import run_experiments, register_env
from agents.GolKenari import GolKenari
from agents.RiskyValley import RiskyValley
from argparse import Namespace

import pickle

class PatchedPPOTrainer(ray.rllib.agents.ppo.PPOTrainer):

    #@override(Trainable)
    def load_checkpoint(self, checkpoint_path: str) -> None:
        extra_data = pickle.load(open(checkpoint_path, "rb"))
        worker = pickle.loads(extra_data["worker"])
        worker = PatchedPPOTrainer.__fix_recursively(worker)
        extra_data["worker"] = pickle.dumps(worker)
        self.__setstate__(extra_data)

    def __fix_recursively(data):
        if isinstance(data, dict):
            return {key: PatchedPPOTrainer.__fix_recursively(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [PatchedPPOTrainer.__fix_recursively(value) for value in data]
        elif data is None:
            return 0
        else:
            return data

class SelfPlay:
    def __init__(self, team, action_lenght):
        # args = Namespace(map="RiskyValley", render=False, gif=False, img=False)
        args = Namespace(map="GolKenariVadisi", render=False, gif=False, img=False)
        agents = [None, "SimpleAgent"]

        self.team = 0
        self.enemy_team = 1
    
        # ray.init(num_gpus=1, log_to_driver=True)
        ray.init()
        # config= {"use_critic": True,
        #      "num_workers": 1,
        #      "use_gae": True,
        #      "lambda": 1.0,
        #      "kl_coeff": 0.2,
        #      "rollout_fragment_length": 200,
        #      "train_batch_size": 4000,
        #      "sgd_minibatch_size": 128,
        #      "shuffle_sequences": True,
        #      "num_sgd_iter": 30,
        #      "lr": 5e-5,
        #      "lr_schedule": None,
        #      "vf_loss_coeff": 1.0,
        #      "framework": "torch",
        #      "entropy_coeff": 0.0,
        #      "entropy_coeff_schedule": None,
        #      "clip_param": 0.3,
        #      "vf_clip_param": 10.0,
        #      "grad_clip": None,
        #      "kl_target": 0.01,
        #      "batch_mode": "truncate_episodes",
        #      "observation_filter": "NoFilter"}
        config= {"use_critic": True,
            "log_level": "WARN",
             "num_workers": 8,
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
        # register_env("ray", lambda config: RiskyValley(args, agents))
        register_env("ray", lambda config: GolKenari(args, agents))
        # ppo_agent = PPOTrainer(config=config, env="ray")
        ppo_agent = PatchedPPOTrainer(config=config, env="ray")
        # ppo_agent = PPOTrainer(env="ray")
        # ppo_agent.restore(checkpoint_path="data/inputs/model/checkpoint_002600/checkpoint-2600") # Modelin Bulunduğu yeri girmeyi unutmayın!
        ppo_agent.restore(checkpoint_path="data/inputs/model/checkpoint_000900/checkpoint-900")
        # ppo_agent.restore(checkpoint_path="models/checkpoint_000005/checkpoint-5") # Modelin Bulunduğu yeri girmeyi unutmayın!
        self.policy = ppo_agent.get_policy()

    def action(self, raw_state):
        '''
        pos=[3, 17]
        target=[10, 15]
        astar(pos,target,state)
        return
        '''
        # state = RiskyValley.just_decode_state(raw_state, self.team, self.enemy_team)
        state = GolKenari.just_decode_state(raw_state, self.team, self.enemy_team)
        actions, _, _ = self.policy.compute_single_action(state.astype(np.float32))
        # location, movement, target, train = RiskyValley.just_take_action(actions, raw_state, self.team)
        location, movement, target, train = GolKenari.just_take_action(actions, raw_state, self.team)
        return (location, movement, target, train)