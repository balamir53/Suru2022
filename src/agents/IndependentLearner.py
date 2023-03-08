import gym
from agents.BaseLearningGym import BaseLearningAgentGym
from ray.rllib.env import MultiAgentEnv
from ray.rllib.examples.env.mock_env import MockEnv
from game import Game
from gym import spaces
import yaml
import numpy as np

def read_hypers():
    with open(f"/workspaces/Suru2022/data/config/RiskyValley.yaml", "r") as f:   
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict

class IndependentLearner(MultiAgentEnv):
    def __init__(self, args, agents, team=0):
        
        self.agents = agents
        self.agentID = 0
        self.dones = set()

        agentos = [None, "RandomAgent"]
        self.game = Game(args, agentos)
        self.train = 0

        self.team = team
        self.enemy_team = 1
        
        self.configs = read_hypers()
        self.height = self.configs['map']['y']
        self.width = self.configs['map']['x']
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.nec_obs = None

        # this has to be defined
        # make it smaller by processing the observation space
        # this is the next step
        self.observation_space = spaces.Box(
            low=-2,
            high=401,
            shape=(2,24*18*10+4),
            dtype=np.int16
        )
        # self.observation_space = spaces.Dict (
        #     {
        #     "observations": spaces.Box(
        #     low=-2,
        #     high=401,
        #     shape=(2,24*18*10+4),
        #     dtype=np.int16
        # ),
        #     # "action_mask" : spaces.Box(0.0, 1.0, shape=self.action_space.shape) }
        #     # "action_mask" : spaces.Box(0, 1, shape=(103,),dtype=np.int8) }
        #     "action_mask" : spaces.Box(0, 1, shape=(49,),dtype=np.int8) }
        # )
        # this is defined action space for just one agent
        self.action_space = spaces.Discrete(7)

        self.resetted = False

        # bu neymis? basta gereksiz bir reward eklemez mi bu
        self.previous_enemy_count = 4
        self.previous_ally_count = 4

    def setup(self, obs_spec, action_spec):
        self.observation_space = obs_spec
        self.action_space = action_spec
        # print("setup")

    def spawn(self):
        # spawn a new agent into the curent episode
        agentID = self.agentID
        # what it this ameka
        # we whould assign en environment to the created
        # agent ?
        # but we want to manage all grouped agents in the 
        # same environment
        # lets continue for a while
        self.agents[agentID] = MockEnv(25)

        self.agentID += 1
        return agentID
    
    def reset(self):
        self.previous_enemy_count = 4
        self.previous_ally_count = 4
        self.episodes += 1
        self.steps = 0

        self.agents = {}
        self.spawn()
        self.resetted = True
        self.dones = set()

        obs = {}
        for i,a in self.agents.items():
            obs[i] = a.reset()
        
        return obs
    
    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        
        # sometimes, add a new agent to the episode
        # this actually will happen when the base will create a new unit
        # keep it for now here
        zono = False
        if zono:
            i = self.spawn()
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)

        # sometimes, kill an existing agent
        # this is also possible for our case
        # actually we have to check for grouping agents first 
        # but lets continue for a while
        zonohe = False
        if zonohe:
            # bring the dead unit id as i
            done[i] = True
            del self.agents[i]

        # didnt get this
        done["__all__"] = len(self.dones) == len(self.agents)
        
        return obs, rew, done, info


