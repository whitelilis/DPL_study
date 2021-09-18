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
