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
        

        # our method resembles the multiagent example in petting zoo
        # agents will be created at the start
        # but we have to figure out a way killing them and spawning new ones
        self.agents = agents
        
        # agent ids are created and handed over via training script?
        # self.agentID = 0
        
        # self.dones = set()

        # creating our game which will run a single environment that will be 
        # played via our agents 
        agentos = [None, "RandomAgent"]
        self.game = Game(args, agentos)

        # parameters for our game
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


        self.observation_space = spaces.Box(
            low=-2,
            high=401,
            shape=(24*18*10+4,),
            dtype=np.int16
        )
        self.action_space = spaces.Discrete(7)
        # this has to be defined
        # make it smaller by processing the observation space
        # this is the next step
        # obs spaces per agent has to be created with identical matrices
        # since truck agents for now have same space
        # will these be defined at the group agent, here or in both
        # check this
        self.observation_spaces = {}
        for x in self.agents:
            self.observation_spaces[x] = spaces.Box(
            low=-2,
            high=401,
            shape=(24*18*10+4,),
            dtype=np.int16
        )
        # self.observation_space = spaces.Box(
        #     low=-2,
        #     high=401,
        #     shape=(2,24*18*10+4),
        #     dtype=np.int16
        # )
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
        self.action_spaces = {}
        for x in self.agents:
            self.action_spaces[x] = spaces.Discrete(7)
        # self.action_space = spaces.Discrete(7)

        # no idea what this is for, keep it for now
        self.resetted = False

        # bu neymis? basta gereksiz bir reward eklemez mi bu
        self.previous_enemy_count = 4
        self.previous_ally_count = 4

    # is this even called?
    def setup(self, obs_spec, action_spec):
        self.observation_space = obs_spec
        self.action_space = action_spec
        # print("setup")
    
    # not used for now
    def spawn(self):
        # spawn a new agent into the curent episode
        agentID = self.agentID
        # what it this ameka
        # we whould assign en environment to the created
        # agent ?
        # but we want to manage all grouped agents in the 
        # same environment
        # lets continue for a while
        # self.agents[agentID] = MockEnv(25)

        self.agentID += 1
        return agentID
    
    def reset(self):
        self.previous_enemy_count = 4
        self.previous_ally_count = 4
        self.episodes += 1
        self.steps = 0

        # consider this in the future
        # self.manipulateMape(self.game.config,self.episodes)

        state = self.game.reset()
        self.nec_obs =state

        # nope
        # self.agents = {}

        # how and when should we use this
        # elaborate
        # self.spawn()
        self.resetted = True
        # self.dones = set()

        # we should usually keep a dictionary 
        # for every agent
        # but because we will use only one environment
        # there actually one observation
        # which will be processed into separate obs for 
        # each agent
        # obs = {}
        # for i,a in self.agents.items():
        #     obs[i] = a.reset()
        
        # because we keep several agents here
        # multiagentenv expects several obs

        # obs = {}
        # for i,a in self.agents:
        #     obs[i] = a.observation_space.sample()
        obs_samples = {}
        for x in self.agents:
            obs_samples[x] = self.observation_spaces[x].sample()
        # return self.observation_space.sample()
        return obs_samples
    
    def step(self, action_dict):
        # wait a little bit
        # self.action_mask = np.ones(49,dtype=np.int8)

        # self.env.step(action[self.env.agent_selection])

        # we are expectin an action dictionary of agents
        action = np.array([x for x in action_dict.values()])
        obs_d = {}
        rew_d = {}
        done_d = {}
        info_d = {}
        while self.env.agents:
            obs, rew, done, info = self.env.last()
            a = self.env.agent_selection
            obs_d[a] = obs
            rew_d[a] = rew
            done_d[a] = done
            info_d[a] = info
            if self.env.dones[self.env.agent_selection]:
                self.env.step(None)
            else:
                break

        all_done = not self.env.agents
        done_d["__all__"] = all_done

        return obs_d, rew_d, done_d, info_d


