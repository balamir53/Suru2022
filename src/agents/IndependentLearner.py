import gym
import math
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
UNITS_PADDING = 50*3 # parameter * (y,x and type)
RESOURCE_PADDING = 50*2 # parameter * (y and x)
TERRAIN_PADDING = 7*7 # parameter
class IndependentLearner(MultiAgentEnv):
    def __init__(self, args, agents, team=0):
        
        # our method resembles the multiagent example in petting zoo
        # agents will be created at the start
        # but we have to figure out a way killing them and spawning new ones
        self.agents = agents
        self.init_agents = copy.copy(agents)
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

        self.load_reward = 0.5
        self.unload_reward = 1

        self.current_action = []

        self.observation_space = spaces.Box(
            low=-40,
            high=401,
            shape=(302,),
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
        self.obs_dict = {}
        self.loads = {}
        self.rewards = {}
        self.dones = {}
        self.infos = {}
        for x in self.agents:
            self.observation_spaces[x] = self.observation_space
            self.obs_dict[x] = []
            self.loads[x] = 0
            self.rewards[x] = 0
            self.dones[x] = False  #if agents die make this True
            self.infos[x] = {}
        self.dones['__all__'] = False
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

        # no idea what this is for, keep it for now
        self.resetted = False

        # bu neymis? basta gereksiz bir reward eklemez mi bu
        self.previous_enemy_count = 4
        self.previous_ally_count = 4

        self.agents_positions = {}
        # get the initial positions of agents
        # should we apply the same logic as in decode state?
        for i in range(len(self.agents)):
            self.agents_positions[self.agents[i]]=(self.configs['blue']['units'][i]['y'], self.configs['blue']['units'][i]['x'])
        self.my_base = (self.configs['blue']['base']['y'],self.configs['blue']['base']['x'])
        
        # gets terrain locs [(location(x,y) ,terrain_type)] --> terrain_type : 'dirt' : 1, 'water' : 2, 'mountain' : 3}
        self.terrain = self.terrain_locs()

        self.old_raw_state = None
        self.firstShot = True

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
        self.agents = copy.copy(self.init_agents)
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
        self.observation_spaces = {}
        self.obs_dict = {}
        self.loads = {}
        self.rewards = {}
        self.dones = {}
        self.infos = {}
        for x in self.agents:
            self.observation_spaces[x] = self.observation_space
            self.obs_dict[x] = []
            self.loads[x] = 0
            self.rewards[x] = 0
            self.dones[x] = False  #if agents die make this True
            self.infos[x] = {}
            self.dones[x] = False

        self.agents_positions = {}
        for i in range(len(self.agents)):
            self.agents_positions[self.agents[i]]=(self.configs['blue']['units'][i]['y'], self.configs['blue']['units'][i]['x'])
        
        # how and when should we use this
        # elaborate
        # self.spawn()
        self.resetted = True
        # self.dones = set()
        self.dones['__all__'] = False

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
    
    def load_reward_check(self, old_load, new_load, truck_id):
        # apply reward for collecting loads
        if new_load > old_load and old_load<3:
            self.rewards[truck_id] += self.load_reward
        # if it unloads the the loads on base
        if self.agents_positions[truck_id] == self.my_base and old_load>new_load:
            self.rewards[truck_id] += self.unload_reward * old_load
        pass

    def  _decode_state(self, obs):
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
        # here we also check loads
        # apply reward for any load increase and decrease if on base

        someone_died = False
        to_be_deleted = []
        for i,x in enumerate(self.agents):
            # check for deaths
            if len(self.agents) != len(my_units):
                someone_died = True
            else:
                someone_died = False
            # check if it is on the tile supposed to be
            move_x, move_y = getMovement(self.agents_positions[x],self.current_action[x])
            new_pos = tuple(map(lambda i, j: i + j, self.agents_positions[x], (move_y, move_x)))
            # check if this in my_units but this still can be wrong
            # maybe another unit moved to the locations and this one cant?
            # or there was a no-go section to go
            old_load = self.loads[x]

            # check for no-go section
            if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.height or new_pos[1] >= self.width:
                new_pos = self.agents_positions[x]
                # don't change position
                self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                self.load_reward_check(old_load, self.loads[x], x)
                if someone_died:
                    dead = True
                    for z in my_units:
                        if z['location'] == new_pos:
                            dead = False
                            break
                    if dead:
                        to_be_deleted.append(x)
                continue

            to_break = False
            # check if there is already a unit set there
            # we are not checking terrain and enemy units
            for y,q in enumerate(self.agents):
                if y == i:
                    continue
                if self.agents_positions[q] == new_pos:
                    to_break = True
                    break
            if to_break:
                # if there is already a unit in that pos
                # set it to old pos
                new_pos = self.agents_positions[x]
                self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                self.load_reward_check(old_load, self.loads[x], x)
                if someone_died:
                    dead = True
                    for z in my_units:
                        if z['location'] == new_pos:
                            dead = False
                            break
                    if dead:
                        to_be_deleted.append(x)
                continue
            am_i_alive = False            
            
            if someone_died or (len(self.agents) == len(my_units) and self.train == 0):
                # two options, either nothing changed
                # or a new unit has been created and another has been killed
                # check self.train
                for z in my_units:
                    # check for the new positions in the new state
                    # maybe i couldnt move there and there is already other agent sitting there?
                    # but we check this above?
                    if z['location'] == new_pos:
                        self.agents_positions[x] = new_pos
                        self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                        self.load_reward_check(old_load, self.loads[x], x) 
                        am_i_alive = True
                        break
            if not am_i_alive and someone_died:
                to_be_deleted.append(x)
                # del self.agents_positions[x]
                # del self.agents_positions_[x]
                # self.agents.remove(x)
                # del self.observation_spaces[x] 
                # del self.obs_dict[x] 
                # del self.loads[x] 
                # del self.rewards[x] 
                # del self.dones[x] 
                # del self.infos[i] 
                continue
            if not am_i_alive:
                # this is wildcard
                # don't change anything for now

                # TODO :
                # check terrain constraints
                
                # it is broken, sth is off, actions are applied maybe in different order
                # chaos amk
                # it may hit an enemy unit ??!!
                new_pos = self.agents_positions[x]
                self.loads[x] = load[self.team][new_pos[0],new_pos[1]]
                self.load_reward_check(old_load, self.loads[x], x)

                # make its done flag true
                # self.dones[x] = True
                pass

        if to_be_deleted:
            for i in range(len(to_be_deleted)):
                del self.agents_positions[to_be_deleted[i]]
                self.agents.remove(to_be_deleted[i])
                del self.observation_spaces[to_be_deleted[i]] 
                del self.obs_dict[to_be_deleted[i]] 
                del self.loads[to_be_deleted[i]] 
                del self.rewards[to_be_deleted[i]] 
                del self.dones[to_be_deleted[i]] 
                # del self.infos[i] 
        counter = 0
        for i, agent in enumerate(self.agents_positions):
            for uni in my_units:
                if self.agents_positions[agent] == uni['location']:
                    counter +=1
        if counter < len(self.agents_positions):
            print('Done')
        # unitss = [*units[0].reshape(-1).tolist(), *units[1].reshape(-1).tolist()]
        # hpss = [*hps[0].reshape(-1).tolist(), *hps[1].reshape(-1).tolist()]
        # basess = [*bases[0].reshape(-1).tolist(), *bases[1].reshape(-1).tolist()]
        # ress = [*res.reshape(-1).tolist()]
        # loads = [*load[0].reshape(-1).tolist(), *load[1].reshape(-1).tolist()]
        # terr = [*terrain.reshape(-1).tolist()]

        # other units and  our base relative distance vectors ( 2 * (#our base + all units-1) + type) (3) 
        #     !! use padding for max units size like 50 (fixed to 50 * 3 = 150)


        # return a dict of agents obs
        for i,x in enumerate(self.agents):
            rel_dists = []
            # add rel dist to base
            my_pos = self.agents_positions[x]
            rel_dists+= (np.array(self.my_base)- np.array(my_pos)).tolist()
            rel_dists.append(self.unit_type['base'])
            # rel dist to friendly units
            for y in my_units:
                # if itself, skip
                if my_pos == y['location']:
                    continue                
                rel_dists+=(np.array(y['location'])- np.array(my_pos)).tolist()
                rel_dists.append(self.unit_type['friend'])
            # rel dist to enemy units
            for y in enemy_units:
                rel_dists+=(np.array(y['location'])- np.array(my_pos)).tolist()
                rel_dists.append(self.unit_type['foe'])
            # add padding for the rest
            rel_dists+=[0]*(UNITS_PADDING-len(rel_dists))

            # check rel distances to resources and take first 50 into account
            res_dists = []
            sorted_dist = []
            for z,y in enumerate(resources):
                # get the distances and indexes as tuple
                dis = int(np.linalg.norm(np.array(y)-np.array(my_pos)))
                sorted_dist.append((dis,z))
            # sort the distances with the indexed
            sorted_dist = sorted(sorted_dist, key= lambda x: x[0])
            # get relative distances of the first RESOURCE_PADDING/2
            index = 0
            if len(sorted_dist)>RESOURCE_PADDING/2:
                index = int(RESOURCE_PADDING/2)
            else: 
                index = len(sorted_dist)
            for z in range(index):
                res_dists+= (np.array(resources[sorted_dist[z][1]])- np.array(my_pos)).tolist()
            # pad the remaining res distances if its shorter than RESOURCE PADDING
            if len(res_dists)<RESOURCE_PADDING:
                res_dists+=[0]*(RESOURCE_PADDING-len(res_dists))
            
            # get the terraing around the agent
            agent_surround = [0] * TERRAIN_PADDING 
            if self.terrain:
                # agent_pos = np.array(self.agents_positions[i])
                counter = 0
                # check for surround terrain in TERRAIN_PADDING box, if there is change agent surround index accordingly with terrain type
                index = int(math.sqrt(TERRAIN_PADDING)//2)
                for hor in range(-index, index+1):
                    for ver in range(-index, index+1):
                        coor = (my_pos[0] + hor, my_pos[1] + ver)
                        lookat = coor[0]*self.width + coor[1]
                        if self.terrain.get(lookat):
                            agent_surround[counter] = self.terrain[lookat]
                        counter += 1
            my_state = (*list(my_pos), self.loads[x], *rel_dists, *res_dists, *agent_surround)
            self.obs_dict[x] = np.array(my_state, dtype=np.int16)
        # state = (*score.tolist(), turn, max_turn, *unitss, *hpss, *basess, *ress, *loads, *terr)
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
        # return np.array(state, dtype=np.int16), (x_max, y_max, my_units, enemy_units, resources, my_base,enemy_base)
        return self.obs_dict, (x_max, y_max, my_units, enemy_units, resources, my_base,enemy_base)
    
    @staticmethod
    def unit_dicts(obs, allies, enemies,  team):
        """This method creates unit dicts to be used in nearest enemy locs."""
        #from the state(obs), following base and dead parameters comes as they are part of a unit. Resources are added just in case.
        unitTagToString = {1: "Truck",2: "LightTank",3: "HeavyTank",4: "Drone",6: "Base",8: "Dead",9: "Resource"}
        lists = [[], []]
        ally_units = obs['units'][team]
        enemy_units = obs['units'][(team+1) % 2]
        #creates a list consisting unit types of both sides.
        units_types = [[],[]]
        for i,x in enumerate(allies):
            units_types[0].append(ally_units[x[0]][x[1]])
        for x in enemies:
            units_types[1].append(enemy_units[x[0]][x[1]])
        # units_types = [[ally_units[ally[0], ally[1]] for ally in allies], [enemy_units[enemy[0], enemy[1]] for enemy in enemies]]
        #creates a list consisting unit locations of both sides.
        unit_locations = [allies, enemies]
        #creates a dict for each side consisting unit type tags and unit locations.
        for i,x in enumerate(unit_locations[0]):
            unit = {
                    "tag" : unitTagToString[units_types[0][i]],
                    "location" : tuple(x)
                    }
            lists[0].append(unit)
        for i in range(len(units_types[1])):
            unit = {
                "tag" : unitTagToString[units_types[1][i]],
                "location" : tuple(unit_locations[1][i])
                }
            lists[1].append(unit)
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
    
    def terrain_locs(self):
        terrain_type = {'d' : 1, 'w' : 2, 'm' : 3}
        if not self.configs['map'].get('terrain'):
            return
        terrain = self.configs['map']['terrain']
        x_max, y_max = self.configs['map']['x'], self.configs['map']['y']
        ter_locs = {}
        for i in range(y_max):
            for j in range(x_max):
               if terrain[i][j] == 'd' or terrain[i][j] == 'w' or terrain[i][j] == 'm':
                    ter_locs[i*self.width+j]=terrain_type[terrain[i][j]]
        return ter_locs
    
    def apply_action(self, action, raw_state, team):
        # this function takes the output of the model and converts it into a reasonable output for the game to play

        # this is specific order as in self.agents
        movement = action[0:7]
        movement = movement.tolist()
        # target = action[7:14]
        # train = action[14]
        
        # target = []

        enemy_order = []

        # here it changes the units order by creating a set but as long as it keeps consistent no problem
        # but action list is handed by the model by the name of the single agents
        # we have to keep track or we can take directly agents position
        
        allies_ = ally_locs(raw_state, team)
        if not self.firstShot:
            allies__ = ally_locs(self.old_raw_state, team)
        self.firstShot = False
        # this is updated in _decode_state after each game step
        allies = self.agents_positions.values()
        
        counter = 0
        for agent11 in allies_:
            for uni11 in allies:
                if agent11 == uni11:
                    counter +=1
        if counter != len(allies):
            print('Hay amk')
        
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

        elif len(allies) > 7:
            ally_count = 7
            locations = allies
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

        locations = list(map(tuple, locations))
        enemy_order = list(map(tuple, enemy_order))

        # TODO : delete this
        self.old_raw_state = raw_state
        

        # this has to be returned in this order according to challenge rules
        return [locations, movement, enemy_order, self.train]
    def step(self, action_dict):
        # wait a little bit
        # self.action_mask = np.ones(49,dtype=np.int8)
        
        # dont forget to reset reward
        for x in self.rewards:
            self.rewards[x] = 0
        
        self.current_action = action_dict

        harvest_reward, kill_reward, martyr_reward = 0, 0, 0

        # self.env.step(action[self.env.agent_selection])

        # we are expectin an action dictionary of agents
        action = np.array([x for x in action_dict.values()])
        action = self.apply_action(action, self.nec_obs, self.team)
        next_state, _, done =  self.game.step(action)

        # we have to update our agents and agents_locations list
        # immediately
        obs_d, next_info = self._decode_state(next_state)
        # _, info = self._decode_state(self.nec_obs)
        # here we will convert the action space
        # into the game inputs as location, movement, target and train
        # game step returns next_state,reward,done

        # obs_d = {} # we got these from _decode_state
        rew_d = self.rewards

        # get rewards
        # we managed this in decode state
        
        if done: #this comes from game step
            for x in self.agents:
                self.dones[x] = True
            self.dones['__all__'] = True

        done_d = self.dones
        info_d = self.infos

        self.nec_obs = next_state

        return obs_d, rew_d, done_d, info_d


