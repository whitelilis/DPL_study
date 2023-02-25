import torch
import torch.nn as nn
import torch.nn.functional as F

# globals
metric_num = len(['o', 'h', 'l', 'c', 'v_delta', 'cash', 'margin', 'profile'])
step_num = 20
time_scales = len(['tick', 'min', '5min', '15min', 'hour', 'day'])
action_count = len(['b1', 's1', 'h', 'c'])
fake_tick = torch.randn((time_scales, step_num, metric_num))
kernel_h = 5 # segment_range
kernel_w = 3 # metric_kinds




class WizardDQN(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """
    def __init__(self):
        super(WizardDQN, self).__init__()

        self.cons = nn.Sequential(
            nn.Conv2d(time_scales, time_scales, (kernel_h, kernel_w), padding=1),
            nn.ReLU(),
            nn.Conv2d(time_scales, time_scales, (kernel_h, kernel_w), padding=1),
            nn.ReLU(),
            nn.Conv2d(time_scales, time_scales, (kernel_h, kernel_w), padding=1),
            nn.ReLU()
        )
        in_w = len(self.cons(fake_tick.unsqueeze(0)).view(-1))

        self.lins = nn.Sequential(
            nn.Linear(in_w, in_w),
            nn.ReLU(),
            nn.Linear(in_w, in_w),
            nn.ReLU(),
            nn.Linear(in_w, action_count),
        )

    def forward(self, x):
        con_out = self.cons(x.unsqueeze(0))
        lin_out = self.lins(con_out.view(-1))
        return F.softmax(lin_out)

import random

class solver():
    def __init__(self, exploration_rate, lr):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.exploration_rate = exploration_rate
        self.dqn = WizardDQN()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.l1 = nn.SmoothL1Loss().to(self.device)
    def act(self, state):
        """Epsilon-greedy action"""
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).cpu()

    def train(self):
        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error




if __name__ == '__main__':
    m1 = WizardDQN()
    print(m1(fake_tick))