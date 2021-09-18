from json import JSONEncoder

import pandas
import enum
import jsonpickle
import torch

#from pys.Tenv import Tenv

from abc import abstractmethod


class Tenv:
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def invalidate_action_probabilities_adjust(self, action_probabilities):
        pass

    @abstractmethod
    def sample_actions(self):
        pass



INIT_PROFILE = 200000

class Direction:
    LONG = 1
    SHORT = -1

class Position:
    def __init__(self, direction: Direction, volume: int, in_price: float):
        self.direction = direction
        self.volume = volume
        self.in_price = in_price
        self.margin: float = 0.1 * in_price
        self.commission: float = 10
        self.mulitiple: float = 10


class Stat():
    def __init__(self, init_profile: float):
        self.cash : float = init_profile
        self.profile : float = 0.0
        self.positions : [Position] = []
        self.metrics : [float] = []
        self.pre_net = init_profile
        self.last_price = 0.0

    def calculate_profile(self, current_price):
        self.profile = 0.0
        for p in self.positions:
            self.profile += (current_price - p.in_price) * p.direction * p.mulitiple

    def update_status(self, current_price, action: torch.Tensor):
        self.calculate_profile(current_price)
        self.last_price = current_price
        new_net = self.cash + self.profile

        step_reward = new_net - self.pre_net
        self.pre_net = new_net

        action_values = action.tolist()
        # handle action
        # todo: handle margin trans
        if action_values[0] > 0:  # long
            new_position = Position(Direction.LONG, 1, current_price)
            self.positions.append(new_position)
            self.cash -= new_position.commission
        elif action_values[1] > 0:  # short
            new_position = Position(Direction.LONG, 1, current_price)
            self.positions.append(new_position)
            self.cash -= new_position.commission
        elif action_values[2] > 0:  # close
            self.cash -= self.positions[0].commission
            self.cash += self.profile
            self.profile = 0
            self.positions.clear()
        else:
            pass

        # construct result

        money = torch.tensor([self.cash, self.profile])
        metrics = torch.tensor(self.metrics)
        return step_reward, torch.cat([money, metrics]), new_net

class SingleEnv(Tenv):
    def invalidate_action_probabilities_adjust(self, action_probabilities):
        """
        Change impossible actions' probability into zero
        :param action_probabilities: [long, short, close, hold]
        :return: changed [long, short, close, hold]
        """
        if self.stat.positions:
            if self.stat.positions[0].direction == Direction.LONG:
                action_probabilities[0] = 0
            else:
                action_probabilities[1] = 0
        else:  # no position, can't close
            action_probabilities[2] = 0
        return action_probabilities

    def __init__(self, file, symbol):
        self.is_done: bool = False
        self.data: pandas.DataFrame = pandas.read_csv(file)[['day', 'time', symbol]]
        self.data.rename(columns={symbol: 'price'}, inplace=True)
        self.back_data = self.data.copy(deep=True)
        self.stat: Stat = Stat(INIT_PROFILE)
        self.timeline: int = 0

    def sample_actions(self):
        return torch.tensor([0.33, 0.33, 0.34])

    def reset(self):
        self.is_done = False
        self.data = self.back_data.copy(deep=True)
        self.stat = Stat(INIT_PROFILE)
        self.timeline = 0

    def step(self, action):
        """
        Apply action,
        :param action: action is a tensor like [long, short, close, hold], which is already validate and 'softmax'ed
        :return: (reward, status, is_done)
        """
        self.timeline = self.timeline + 1

        if self.timeline >= len(self.data):
            return 0, None, True
        else:
            new_price: float = self.data.iloc[self.timeline]['price']
            reward, status, current_net = self.stat.update_status(new_price, action)  # profile
            # if net is less than 0.8, stop
            if current_net < 0.8 * INIT_PROFILE:
                return reward, status, True
            else:
                return reward, status, False

    def render(self):
        direction = self.stat.positions[0].direction if self.stat.positions else None
        in_price  = self.stat.positions[0].in_price if self.stat.positions else None
        last_price = self.stat.last_price
        print(f"timeline: {self.timeline}, net: {self.stat.pre_net}, cash: {self.stat.cash}, profile: {self.stat.profile}, direction: {direction}, in_price: {in_price}, now_price: {last_price}")


e = SingleEnv('/Users/liuzhe/github/DPL_study/data/snap.csv', 'AP')

one_long = torch.tensor([1, 0, 0, 0])
one_short = torch.tensor([0, 1, 0, 0])
one_close = torch.tensor([0, 0, 1, 0])
one_hold = torch.tensor([0, 0, 0, 1])
x = 3

e.step(one_long)
for _ in range(30):
    e.step(one_hold)
e.render()
    
print("got here 2")

y = 5

x = pandas.read_csv('/Users/liuzhe/github/DPL_study/data/snap.csv')
    
