import gym
import copy
from agents.BaseLearningGym import BaseLearningAgentGym
from ray.rllib.env import MultiAgentEnv
from ray.rllib.examples.env.mock_env import MockEnv
from game import Game
from gym import spaces
import yaml
import numpy as np
# import utilities
from utilities import ally_locs, enemy_locs, nearest_enemy_selective, getMovement

def read_hypers():
    with open(f"/workspaces/Suru2022/data/config/RiskyValley.yaml", "r") as f:   
        hyperparams_dict = yaml.safe_load(f)
        return hyperparams_dict
UNITS_PADDING = 50*3
RESOURCE_PADDING = 50*2
class IndependentLearner(MultiAgentEnv):
    def __init__(self, args, agents, team=0):
        
        # our method resembles the multiagent example in petting zoo
        # agents will be created at the start
        # but we have to figure out a way killing them and spawning new ones
        self.agents = agents
        self.tagToString = {
            1: "Truck",
            2: "LightTank",
            3: "HeavyTank",
            4: "Drone",
        }

        # keep it simple for now
        # self.unit_type = {
        #     'base': 0,
        #     'my_truck': 1,
        #     'my_ltank': 2,
        #     'my_htank': 3,
        #     'my_drone': 4,
        #     'en_truck': 5,
        #     'en_ltank': 6,
        #     'en_htank': 7,
        #     'en_drone': 8
        # }
        self.unit_type = {
            'base': 1,
            'friend': 2,
            'foe': 3
        }
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

        self.current_action = []

        self.observation_space = spaces.Box(
            low=-2,
            high=401,
            shape=(685,),
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
            self.observation_spaces[x] = self.observation_space
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
            self.action_spaces[x] = self.action_space
        # self.action_space = spaces.Discrete(7)

        self.obs_dict = []

        # no idea what this is for, keep it for now
        self.resetted = False

        # bu neymis? basta gereksiz bir reward eklemez mi bu
        self.previous_enemy_count = 4
        self.previous_ally_count = 4

        self.agents_positions = []
        # get the initial positions of agents
        # should we apply the same logic as in decode state?
        for i in range(len(self.agents)):
            self.agents_positions.append((self.configs['blue']['units'][i]['y'], self.configs['blue']['units'][i]['x']))
        self.my_base = (self.configs['blue']['base']['y'],self.configs['blue']['base']['x'])
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
    def _decode_state(self, obs):
        turn = obs['turn']
        max_turn = obs['max_turn'] 
        units = obs['units']
        hps = obs['hps']
        bases = obs['bases']
        score = obs['score']
        res = obs['resources']
        load = obs['loads']
        terrain = obs["terrain"]
        y_max, x_max = res.shape
        my_units = []
        enemy_units = []
        resources = []
        for i in range(y_max):
            for j in range(x_max):
                if units[self.team][i][j]<6 and units[self.team][i][j] != 0:
                    my_units.append(
                    {   
                        'unit': units[self.team][i][j],
                        'tag': self.tagToString[units[self.team][i][j]],
                        'hp': hps[self.team][i][j],
                        'location': (i,j),
                        'load': load[self.team][i][j]
                    }
                    )
                if units[self.enemy_team][i][j]<6 and units[self.enemy_team][i][j] != 0:
                    enemy_units.append(
                    {   
                        'unit': units[self.enemy_team][i][j],
                        'tag': self.tagToString[units[self.enemy_team][i][j]],
                        'hp': hps[self.enemy_team][i][j],
                        'location': (i,j),
                        'load': load[self.enemy_team][i][j]
                    }
                    )
                if res[i][j]==1:
                    resources.append((i,j))
                if bases[self.team][i][j]:
                    my_base = (i,j)
                if bases[self.enemy_team][i][j]:
                    enemy_base = (i,j)
        
        # elaborate
        if self.train > 0:
            self.agents.append()
            self.agents_positions()

        # update here self.agents and self.agents_positions
        # how to check which agent at which position has been killed?
        # print(my_units)
        for i,x in enumerate(self.agents):
            # check if it is on the tile supposed to be
            move_x, move_y = getMovement(self.agents_positions[i],self.current_action['truck'+str(i)])
            new_pos = tuple(map(lambda i, j: i + j, self.agents_positions[i], (move_y, move_x)))
            # check if this in my_units but this still can be wrong
            # maybe another unit moved to the locations and this one cant?
            # or there was a no-go section to go

            # check for no-go section
            if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.height or new_pos[1] >= self.width:
                new_pos = self.agents_positions[i]
                continue

            to_break = False
            # check if there is already a unit set there
            for y in range (0, len(self.agents)):
                if y == i:
                    continue
                if self.agents_positions[y] == new_pos:
                    to_break = True
                    break
            if to_break:
                # if there is already a unit in that pos
                # set it to old pos
                new_pos = self.agents_positions[i]
                continue
            am_i_alive = False
            if len(self.agents) == len(my_units) and self.train == 0:
                # two options, either nothing changed
                # or a new unit has been created and another has been killed
                # check self.train
                for x in my_units:
                    # check for the new positions in the new state
                    if x['location'] == new_pos:
                        self.agents_positions[i] = new_pos
                        am_i_alive = True
                        break
            if not am_i_alive and len(self.agents) != len(my_units):
                del self.agents_positions[i]
                del self.agents[i]
            if not am_i_alive:
                # this is wildcard
                # don't change anything for now
                pass
        unitss = [*units[0].reshape(-1).tolist(), *units[1].reshape(-1).tolist()]
        hpss = [*hps[0].reshape(-1).tolist(), *hps[1].reshape(-1).tolist()]
        basess = [*bases[0].reshape(-1).tolist(), *bases[1].reshape(-1).tolist()]
        ress = [*res.reshape(-1).tolist()]
        loads = [*load[0].reshape(-1).tolist(), *load[1].reshape(-1).tolist()]
        terr = [*terrain.reshape(-1).tolist()]

        # other units and  our base relative distance vectors ( 2 * (#our base + all units-1) + type) (3) 
        #     !! use padding for max units size like 50 (fixed to 50 * 3 = 150)


        # return a dict of agents obs
        for i,x in enumerate(self.agents):
            rel_dists = []
            # add rel dist to base
            rel_dists+= (np.array(self.my_base)- np.array(self.agents_positions[i])).tolist()
            rel_dists.append(self.unit_type['base'])
            # rel dist to friendly units
            for y in my_units:
                # if itself, skip
                if self.agents_positions[i] == y['location']:
                    continue                
                rel_dists+=(np.array(y['location'])- np.array(self.agents_positions[i])).tolist()
                rel_dists.append(self.unit_type['friend'])
            # rel dist to enemy units
            for y in enemy_units:
                rel_dists+=(np.array(y['location'])- np.array(self.agents_positions[i])).tolist()
                rel_dists.append(self.unit_type['foe'])
            # add padding for the rest
            rel_dists+=[0]*(UNITS_PADDING-len(rel_dists))

            # check rel distances to resources and take first 50 into account
            res_dists = []
            sorted_dist = []
            for x,y in enumerate(resources):
                # get the distances and indexes as tuple
                dis = int(np.linalg.norm(np.array(y)-np.array(self.agents_positions[i])))
                sorted_dist.append((dis,x))
            # sort the distances with the indexed
            sorted_dist = sorted(sorted_dist, key= lambda x: x[0])
            # get relative distances of the first RESOURCE_PADDING/2
            index = 0
            if len(sorted_dist)>RESOURCE_PADDING/2:
                index = int(RESOURCE_PADDING/2)
            else: 
                index = len(sorted_dist)
            for x in range(index):
                res_dists+= (np.array(resources[sorted_dist[x][1]])- np.array(self.agents_positions[i])).tolist()
            # pad the remaining res distances if its shorter than RESOURCE PADDING
            if len(res_dists)<RESOURCE_PADDING:
                res_dists+=[0]*(RESOURCE_PADDING-len(res_dists))
                
        state = (*score.tolist(), turn, max_turn, *unitss, *hpss, *basess, *ress, *loads, *terr)
        '''
        state actually turns here into observation space for the model 
        we will decrease it for the truck agent
        current model:
        scores(2) [model doesnt get any reward for these] -not needed
        turn (1) [no model effect] -not needed (we can apply neg rew for delays)
        max_turn (1) [no model effect]
        unitss (map size * 2) - needed, but it can be reshaped as the agent coordinates and relative distance to others?
        hpss (map size * 2)  - not needed for the initial model
        basess (map size * 2) - size too much, can be modeled as coordinates?
        ress (map size) - needed, but it can be reshaped as coordinates as well?
        loads (map size * 2) - we don't need all the loads, can keep only agent truck load?
        terr (map size) - needed
        TOTAL: 4324
        '''
        '''
        ********** new model shape *********
        agents coordinate (2)
        its load (1)
        other units and  our base relative distance vectors ( 2 * (#our base + all units-1) + type) (3) 
            !! use padding for max units size like 50 (fixed to 50 * 3 = 150)
        relative distances to closest (say 50) resources (2 * resources) (100)
        terrain (map size) ? how to manage this ? cant go into water, but we dont apply any neg rew? should we?
            !! we can keep 7x7 grid for the agent staying in the center
        TOTAL: 302
        [has to be calculated per agent, all agents observations will be then handed to the model as separate obs]
        [we need also calculate reward per agent]
        '''
        return np.array(state, dtype=np.int16), (x_max, y_max, my_units, enemy_units, resources, my_base,enemy_base)
    
    @staticmethod
    def unit_dicts(obs, allies, enemies,  team):
        tagToString = {
            1: "Truck",
            2: "LightTank",
            3: "HeavyTank",
            4: "Drone",
        }
        changed = 0
        lists = [[], []]
        ally_units = obs['units'][team]
        enemy_units = obs['units'][(team+1) % 2]
        units_types = [[ally_units[ally[0], ally[1]] for ally in allies], [enemy_units[enemy[0], enemy[1]] for enemy in enemies]]
        unit_locations = [allies, enemies]
        for index in range(2):
            for i in range(len(units_types[index])):
                if units_types[index][i] > 4:
                    units_types[index][i] = 4
                    changed += 1
                unit = {
                    "tag" : tagToString[units_types[index][i]],
                    "location" : tuple(unit_locations[index][i])
                }
                lists[index].append(unit)
        return lists[0], lists[1] 
    
    def nearest_enemy_details(allies, enemies):
            nearest_enemy_detail = []
            for ally in allies:
                # if the ally unit is a truck, append none to nearest enemy list since it is not a fire element and continue to new ally unit.
                if ally["tag"] == "Truck":
                    nearest_enemy_detail.append(None)
                    continue
                if len(enemies) == 0 or len(enemies) < 0:
                    break
                nearest_enemy_detail.append(nearest_enemy_selective(ally, enemies))
            return nearest_enemy_detail
    
    def nearest_enemy_list(nearest_enemy_dict):
        nearest_enemy_locs = []
        for n_enemy in nearest_enemy_dict:
            if n_enemy is None:
                nearest_enemy_locs.append(np.asarray([3, 0]))
                continue
            nearest_enemy_locs.append(np.asarray(list(n_enemy["location"])))
        return nearest_enemy_locs
    
    def apply_action(self, action, raw_state, team):
        # this function takes the output of the model
        # and converts it into a reasonable output for
        # the game to play

        # this is specific order as in self.agents
        movement = action[0:7]
        movement = movement.tolist()
        # target = action[7:14]
        # train = action[14]
        
        # target = []

        enemy_order = []

        # here it changes the units order
        # by creating a set
        # but as long as it keeps consistent no problem
        # but action list is handed by the model
        # by the name of the single agents
        # we have to keep track
        # or we can take directly agents position
        # allies = ally_locs(raw_state, team)
        # this is updatep in _decode_state after each game step
        allies = copy.copy(self.agents_positions)
        # but how to check if our agent has been killed
        # or a new unit has been created (maybe we can use self.train)
        # but it has to be controlled immediately after game step
        # not here, in _decode_state
        enemies = enemy_locs(raw_state, team)
        my_unit_dict, enemy_unit_dict = IndependentLearner.unit_dicts(raw_state, allies, enemies, team)  
              
        nearest_enemy_dict = IndependentLearner.nearest_enemy_details(my_unit_dict, enemy_unit_dict)
        nearest_enemy_locs = IndependentLearner.nearest_enemy_list(nearest_enemy_dict)
                
        if 0 > len(allies):
            print("Neden negatif adamların var ?")
            raise ValueError
        elif 0 == len(allies):
            locations = []
            movement = []
            target = []
            return [locations, movement, target, self.train]
        elif 0 < len(allies) <= 7:
            ally_count = len(allies)
            locations = allies

            # counter = 0
            # for j in target: 
            #     if len(enemies) == 0:
            #         # yok artik alum
            #         enemy_order = [[3, 0] for i in range(ally_count)]
            #         continue
            #     k = j % len(enemies)
            #     if counter == ally_count:
            #         break
            #     if len(enemies) <= 0:
            #         break
            #     enemy_order.append(enemies[k].tolist())
            #     counter += 1

            ##added by luchy: this part creates a list of closest enemy order. If num of enemies == 0 creates a dummy fire point for each ally.
            if len(enemies) == 0:
                    # yok artik alum
                enemy_order = [[3, 0] for i in range(ally_count)]
            else:
                enemy_order = copy.copy(nearest_enemy_locs)

            while len(enemy_order) > ally_count:
                enemy_order.pop()
            while len(movement) > ally_count:
                # extracting the unused movement parameters
                # there are seven values by default
                # these actions have to be masked
                movement.pop()
        
        # mask out the unused part of the action space
        # This have to be set at the beginning
        # or no
        # This value changes throughout the sim
        # self.action_mask[]

        
        elif len(allies) > 7:
            ally_count = 7
            locations = allies

            # counter = 0
            # for j in target:
            #     if len(enemies) == 0:
            #         # bu ne oluyor press tv
            #         enemy_order = [[3, 0] for i in range(ally_count)]
            #         continue
            #     k = j % len(enemies)
            #     if counter == ally_count:
            #         break
            #     if len(enemies) <= 0:
            #         break
            #     enemy_order.append(enemies[k].tolist())
            #     counter += 1
            ##added by luchy:
            if len(enemies) == 0:
                    # yok artik alum
                enemy_order = [[3, 0] for i in range(ally_count)]
            else:
                enemy_order = copy.copy(nearest_enemy_locs)
            
            ##added by luchy:due to creating nearest enemy locs for each ally, if number of allies are over 7, only 7 targets must be defined.
            enemy_order = enemy_order[:7]
            
            while len(locations) > 7:
                locations = list(locations)[:7]

        # bu nedir, manuel trucklara 0 atama, yanlis
        # movement = multi_forced_anchor(movement, raw_state, team)

        if len(locations) > 0:
            locations = list(map(list, locations))
        
        # boyle bisi olabilir mi ya
        # locations'dan biri, bir düşmana 2 adımda veya daha yakınsa dur (movement=0) ve ona ateş et (target = arg.min(distances))
        # for i in range(len(locations)):
        #     for k in range(len(enemy_order)):
        #         if getDistance(locations[i], enemy_order[k]) <= 3:
        #             movement[i] = 0
        #             enemy_order[i] = enemy_order[k]

        # also a model manipulation, prevents model learning that actually
        ##added by luchy:by this if the distance between ally and enemy is less than 3 then movement will be 0 as a preparation to shoot.
        # for i in range(len(locations)):
        #     if getDistance(locations[i], enemy_order[i]) <= 3:
        #         movement[i] = 0

        locations = list(map(tuple, locations))
        enemy_order = list(map(tuple, enemy_order))

        # this has to be returned in this order according to challenge rules
        return [locations, movement, enemy_order, self.train]
    def step(self, action_dict):
        # wait a little bit
        # self.action_mask = np.ones(49,dtype=np.int8)

        self.current_action = action_dict

        harvest_reward, kill_reward, martyr_reward = 0, 0, 0

        # self.env.step(action[self.env.agent_selection])

        # we are expectin an action dictionary of agents
        action = np.array([x for x in action_dict.values()])
        action = self.apply_action(action, self.nec_obs, self.team)
        next_state, _, done =  self.game.step(action)

        # we have to update our agents and agents_locations list
        # immediately
        next_state_obs, next_info = self._decode_state(next_state)
        # _, info = self._decode_state(self.nec_obs)
        # here we will convert the action space
        # into the game inputs as location, movement, target and train
        # game step returns next_state,reward,done

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


