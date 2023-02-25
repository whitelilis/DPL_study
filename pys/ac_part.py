#from https://mathpretty.com/12665.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticModel(nn.Module):
    def __init__(self):
        super(ActorCriticModel, self).__init__()
        self.fc1 = nn.Linear(48, 24)
        self.fc2 = nn.Linear(24, 12)
        self.action = nn.Linear(12, 4)
        self.value = nn.Linear(12, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action(x), dim=-1)
        state_values = self.value(x)
        return action_probs, state_values


def trainIters(env, ActorCriticModel, num_episodes, gamma = 0.9):
    optimizer = torch.optim.Adam(ActorCriticModel.parameters(), 0.03) # 注意学习率的大小
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    # 记录reward和总长度的变化
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes+1),
        episode_rewards=np.zeros(num_episodes+1))
    for i_episode in range(1, num_episodes+1):
        # 开始一轮游戏
        state = env.reset() # 环境重置
        state = get_screen(state) # 将state转换为one-hot的tensor, 用作网络的输入.
        log_probs = []
        rewards = []
        state_values = []
        for t in itertools.count():
            action_probs, state_value = ActorCriticModel(state.squeeze(0)) # 返回当前state下不同action的概率
            action = torch.multinomial(action_probs, 1).item() # 选取一个action
            log_prob = torch.log(action_probs[action])
            next_state, reward, done, _ = env.step(action) # 获得下一个状态
            # 计算统计数据
            stats.episode_rewards[i_episode] += reward # 计算累计奖励
            stats.episode_lengths[i_episode] = t # 查看每一轮的时间
            # 将值转换为tensor
            reward = torch.tensor([reward], device=device)
            next_state_tensor = get_screen(next_state)
            # 将信息存入List
            log_probs.append(log_prob.view(-1))
            rewards.append(reward)
            state_values.append(state_value)
            # 状态更新
            state = next_state_tensor
            if done: # 当一轮结束之后, 开始更新
                returns = []
                Gt = 0
                pw = 0
                # print(rewards)
                for reward in rewards[::-1]:
                    Gt = Gt + (gamma ** pw) * reward # 写成Gt += (gamma ** pw) * reward, 最后returns里东西都是一样的
                    # print(Gt)
                    pw += 1
                    returns.append(Gt)
                returns = returns[::-1]
                returns = torch.cat(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                # print(returns)
                log_probs = torch.cat(log_probs)
                state_values = torch.cat(state_values)
                # print(returns)
                # print(log_probs)
                # print(state_values)
                advantage = returns.detach() - state_values
                critic_loss = F.smooth_l1_loss(state_values, returns.detach())
                actor_loss = (-log_probs * advantage.detach()).mean()
                loss = critic_loss + actor_loss
                # 更新critic
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Episode: {}, total steps: {}'.format(i_episode, t))
                if t>20:
                    scheduler.step()
                break
    return stats