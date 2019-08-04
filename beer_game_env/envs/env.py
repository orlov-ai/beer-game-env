import cloudpickle
import gym
from gym import error, spaces
from gym.utils import seeding
import itertools
from collections import deque
import numpy as np


def add_noise_to_init(init, noise):
    """
    Add noise to initial values.
    :type init: iterable, list or (list of lists)
    :type noise: np.array, 1-dimensional
    :rtype with_noise: np.array or (list of np.arrays)
    """
    # TODO add continuous variant
    is_init_array = all([isinstance(x, (float, int)) for x in init])

    if is_init_array:  # init is a list
        with_noise = (np.array(init) + noise).astype(int).tolist()
    else:  # init is a lists of lists
        with_noise = []
        c = 0
        for row in init:
            noise_row = np.array(row) + noise[c:(c + len(row))]
            noise_row = noise_row.astype(int).tolist()
            c += len(noise_row)
            with_noise.append(noise_row)

    return with_noise


def get_init_len(init):
    """
    Calculate total number of elements in a 1D array or list of lists.
    :type init: iterable, list or (list of lists)
    :rtype: int
    """
    is_init_array = all([isinstance(x, (float, int, np.int64)) for x in init])
    if is_init_array:
        init_len = len(init)
    else:
        init_len = len(list(itertools.chain.from_iterable(init)))
    return init_len


def transform_obs(x: dict):
    """
    transform dict of observations (one step) to an array
    :param x: dict
    :rtype: np.array
    """
    return np.array((x['next_incoming_order'], x['current_stock'], x['cum_cost'], *x['shipments'], *x['orders']))


def state_dict_to_array(state_dict: dict):
    """
    transform dict of observations (current step and previous steps) to an array
    :param state_dict:
    :rtype: np.array
    """
    # todo in this state this function is not use, need to use it
    current_obs = transform_obs(state_dict)
    if 'prev' in state_dict:
        prev_obs = np.hstack([transform_obs(x) for x in state_dict['prev']])
        flatten = np.hstack((current_obs, prev_obs))
    else:
        flatten = current_obs
    return flatten


class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents: int, env_type: str, n_turns_per_game=20,
                 add_noise_initialization=False, seed=None):
        super().__init__()
        self.orders = []
        self.shipments = []
        self.next_incoming_orders = []
        self.stocks = []
        self.holding_cost = None
        self.stockout_cost = None
        self.cum_holding_cost = None
        self.cum_stockout_cost = None
        self.turns = None
        self.score_weight = None
        self.turn = None
        self.done = True
        self.n_states_concatenated = None
        self.prev_states = None
        self.np_random = None

        self.n_agents = n_agents
        self.env_type = env_type
        if self.env_type not in ['classical', 'uniform_0_2', 'normal_10_4']:
            raise NotImplementedError("env_type must be in ['classical', 'uniform_0_2', 'normal_10_4']")

        self.n_turns = n_turns_per_game
        self.add_noise_initialization = add_noise_initialization
        self.seed(seed)

        # TODO calculate state shape
        #self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

    def _save(self):
        """
        serialize environment to a pickle string
        :rtype: string
        """
        canned = cloudpickle.dumps(self)
        return canned

    def _load(self, pickle_string):
        """
        deserialize environment from a pickle string
        """
        self.__dict__.update(cloudpickle.loads(pickle_string).__dict__)

    def _get_observations(self):
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            observations[i] = {'current_stock': self.stocks[i], 'turn': self.turn,
                               'cum_cost': self.cum_holding_cost[i] + self.cum_stockout_cost[i],
                               'shipments': list(self.shipments[i]), 'orders': list(self.orders[i])[::-1],
                               'next_incoming_order': self.next_incoming_orders[i]}
        return observations

    def _get_rewards(self):
        return -(self.holding_cost + self.stockout_cost)

    def _get_demand(self):
        return self.turns[self.turn]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.done = False

        if self.env_type == 'classical':
            temp_orders = [[4, 4]] * (self.n_agents - 1) + [[4]]
            temp_shipments = [[4, 4]] * self.n_agents
            self.next_incoming_orders = [4] * self.n_agents
            self.stocks = [12] * self.n_agents

            if self.add_noise_initialization:
                # noise is uniform [-2,2]
                orders_noise = np.random.choice(np.arange(5), size=get_init_len(temp_orders)) - 2
                temp_orders = add_noise_to_init(temp_orders, orders_noise)

                shipments_noise = np.random.choice(np.arange(5), size=get_init_len(temp_shipments)) - 2
                temp_shipments = add_noise_to_init(temp_shipments, shipments_noise)

                last_incoming_orders_noise = np.random.choice(np.arange(5),
                                                              size=get_init_len(self.next_incoming_orders)) - 2
                self.next_incoming_orders = add_noise_to_init(self.next_incoming_orders, last_incoming_orders_noise)

                stocks_noise = np.random.choice(np.arange(13), size=get_init_len(self.stocks)) - 6
                self.stocks = add_noise_to_init(self.stocks, stocks_noise)

            self.turns = [4] * 4 + [8] * (self.n_turns - 4)
            self.score_weight = [[0.5] * self.n_agents, [1] * self.n_agents]

        elif self.env_type == 'uniform_0_2':
            temp_orders = [[1, 1]] * (self.n_agents - 1) + [[1]]
            temp_shipments = [[1, 1]] * self.n_agents
            self.next_incoming_orders = [1] * self.n_agents
            self.stocks = [4] * self.n_agents

            if self.add_noise_initialization:
                # noise is uniform [-1,1]
                orders_noise = np.random.choice(np.arange(3), size=get_init_len(temp_orders)) - 1
                temp_orders = add_noise_to_init(temp_orders, orders_noise)

                shipments_noise = np.random.choice(np.arange(3), size=get_init_len(temp_shipments)) - 1
                temp_shipments = add_noise_to_init(temp_shipments, shipments_noise)

                last_incoming_orders_noise = np.random.choice(np.arange(3),
                                                              size=get_init_len(self.next_incoming_orders)) - 1
                self.next_incoming_orders = add_noise_to_init(self.next_incoming_orders, last_incoming_orders_noise)

                stocks_noise = np.random.choice(np.arange(5), size=get_init_len(self.stocks)) - 2
                self.stocks = add_noise_to_init(self.stocks, stocks_noise)

            # uniform [0, 2]
            self.turns = self.np_random.uniform(low=0, high=3, size=self.n_turns).astype(np.int)
            self.score_weight = [[0.5] * self.n_agents, [1] * self.n_agents]

        elif self.env_type == 'normal_10_4':
            temp_orders = [[10, 10]] * (self.n_agents - 1) + [[10]]
            temp_shipments = [[10, 10]] * self.n_agents
            self.next_incoming_orders = [10] * self.n_agents
            self.stocks = [40] * self.n_agents

            if self.add_noise_initialization:
                # noise is uniform [-1,1]
                orders_noise = np.random.normal(loc=0, scale=5, size=get_init_len(temp_orders))
                orders_noise = np.clip(orders_noise, -10, 10)  # clip to prevent negative orders
                temp_orders = add_noise_to_init(temp_orders, orders_noise)

                shipments_noise = np.random.normal(loc=0, scale=5, size=get_init_len(temp_shipments))
                shipments_noise = np.clip(shipments_noise, -10, 10)  # clip to prevent negative shipments
                temp_shipments = add_noise_to_init(temp_shipments, shipments_noise)

                last_incoming_orders_noise = np.random.normal(loc=0,
                                                              scale=5, size=get_init_len(self.next_incoming_orders))
                last_incoming_orders_noise = np.clip(last_incoming_orders_noise, -10, 10)
                self.next_incoming_orders = add_noise_to_init(self.next_incoming_orders, last_incoming_orders_noise)

                stocks_noise = np.random.normal(loc=0, scale=4, size=get_init_len(self.stocks))
                stocks_noise = np.clip(stocks_noise, -10, 10)
                self.stocks = add_noise_to_init(self.stocks, stocks_noise)

            self.turns = self.np_random.normal(loc=10, scale=4, size=self.n_turns)
            self.turns = np.clip(self.turns, 0, 1000).astype(np.int)
            # dqn paper page 24
            self.score_weight = [[1.0, 0.75, 0.5, 0.25] * self.n_agents, [10.0] + [0.0] * (self.n_agents - 1)]

        else:
            raise ValueError('wrong env_type')

        # initialize other variables
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.orders = [deque(x) for x in temp_orders]
        self.shipments = [deque(x) for x in temp_shipments]
        self.turn = 0
        self.done = False

        self.n_states_concatenated = 3
        temp_obs = [None] * self.n_agents
        for i in range(self.n_agents):
            temp_obs[i] = {'current_stock': self.stocks[i], 'turn': self.turn,
                           'cum_cost': self.cum_holding_cost[i] + self.cum_stockout_cost[i],
                           'shipments': list(self.shipments[i]), 'orders': list(self.orders[i])[::-1],
                           'next_incoming_order': self.next_incoming_orders[i]}
        prev_state = temp_obs
        self.prev_states = deque([prev_state] * (self.n_states_concatenated - 1))
        return self._get_observations()

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        print('\n' + '=' * 20)
        print('Turn:     ', self.turn)
        print('Stocks:   ', ", ".join([str(x) for x in self.stocks]))
        print('Orders:   ', [list(x) for x in self.orders])
        print('Shipments:', [list(x) for x in self.shipments])
        print('Last incoming orders:  ', self.next_incoming_orders)
        print('Cum holding cost:  ', self.cum_stockout_cost)
        print('Cum stockout cost: ', self.cum_holding_cost)
        print('Last holding cost: ', self.holding_cost)
        print('Last stockout cost:', self.stockout_cost)

    def step(self, action: list):
        # sanity checks
        if self.done:
            raise error.ResetNeeded('Environment is finished, please run env.reset() before taking actions')
        if get_init_len(action) != self.n_agents:
            raise error.InvalidAction(f'Length of action array must be same as n_agents({self.n_agents})')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")

        # concatenate previous states, self.prev_states in an queue of previous states
        self.prev_states.popleft()
        self.prev_states.append(self._get_observations())
        # make incoming step
        demand = self._get_demand()
        orders_inc = [order.popleft() for order in self.orders]
        self.next_incoming_orders = [demand] + orders_inc[:-1]
        ship_inc = [shipment.popleft() for shipment in self.shipments]
        # calculate shipments respecting orders and stock levels
        for i in range(self.n_agents - 1):
            max_possible_shipment = max(0, self.stocks[i + 1]) + ship_inc[i + 1]  # stock + incoming shipment
            order = orders_inc[i] + max(0, -self.stocks[i + 1])  # incoming order + stockout
            max_possible_shipment = min(order, max_possible_shipment)
            self.shipments[i].append(max_possible_shipment)
        self.shipments[-1].append(orders_inc[-1])
        # update stocks
        self.stocks = [(stock + inc) for stock, inc in zip(self.stocks, ship_inc)]
        for i in range(1, self.n_agents):
            self.stocks[i] -= orders_inc[i - 1]
        self.stocks[0] -= demand
        # update orders
        for i in range(self.n_agents):
            self.orders[i].append(action[i])
        self.next_incoming_orders = [self._get_demand()] + [x[0] for x in self.orders[:-1]]

        # calculate costs
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        for i in range(self.n_agents):
            if self.stocks[i] >= 0:
                self.holding_cost[i] = self.stocks[i] * self.score_weight[0][i]
            else:
                self.stockout_cost[i] = -self.stocks[i] * self.score_weight[1][i]
        self.cum_holding_cost += self.holding_cost
        self.cum_stockout_cost += self.stockout_cost
        # calculate reward
        rewards = self._get_rewards()

        # check if done
        if self.turn == self.n_turns - 1:
            self.done = True
        else:
            self.turn += 1
        state = self._get_observations()
        # todo flatten observation dict
        return state, rewards, self.done, {}


if __name__ == '__main__':
    env = BeerGame(n_agents=4, env_type='classical')
    start_state = env.reset()
    for i, obs in enumerate(start_state):
        print(f'Agent {i} observation: {obs}')
    env.render()
    done = False
    while not done:
        actions = np.random.uniform(0, 16, size=4)
        actions = actions.astype(int)
        step_state, step_rewards, done, _ = env.step(actions)
        env.render()

    # you can also save and load environment via
    # canned_env = env._save()
    # env._load(canned_env)
