import torch
import torch.nn as nn
import torch.nn.functional as F

# globals
metric_num = len(['o', 'h', 'l', 'c', 'v_delta',])
step_num = 20
time_scales = len(['tick', 'min', '5min', '15min', 'hour', 'day'])
action_count = len(['b1', 's1', 'h', 'c'])
fake_tick = torch.randn((time_scales, step_num, metric_num))
kernel_h = 5 # segment_range
kernel_w = 3 # metric_kinds




class WizardSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """
    def __init__(self):
        super(WizardSolver, self).__init__()

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
            nn.Linear(in_w, in_w),
            nn.Linear(in_w, action_count),
        )

    def forward(self, x):
        con_out = self.cons(x.unsqueeze(0))
        lin_out = self.lins(con_out.view(-1))
        return F.softmax(lin_out)


if __name__ == '__main__':
    m1 = WizardSolver()
    print(m1(fake_tick))