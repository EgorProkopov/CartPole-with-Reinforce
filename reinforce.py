import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.optim as optim

import gym


gamma = 0.99


class Strategy(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Strategy, self).__init__()

        self.model = nn.Sequential(*[
            nn.Linear(in_dim, 64),
            nn.ReLU(),

            nn.Linear(64, out_dim)
        ])
        self.onpolicy_reset()
        self.train()


    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []


    def forward(self, x):
        pdparam = self.model(x)
        return pdparam


    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))
        pdparam = self.forward(x)

        pd = distributions.Categorical(logits=pdparam)
        action = pd.sample()

        log_prob = pd.log_prob(action)
        self.log_probs.append(log_prob)

        return action.item()


    
def train(strategy, optimizer):
    # ret calculation
    T = len(strategy.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0

    for t in reversed(range(T)):
        future_ret = strategy.rewards[t] + gamma*future_ret
        rets[t] = future_ret

    # loss calculation
    rets = torch.tensor(rets)
    log_probs = torch.stack(strategy.log_probs)
    loss = -log_probs * rets
    loss = torch.sum(loss)

    # strategy train    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


def main():
    env = gym.make('CartPole-v0')
    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    strategy = Strategy(in_dim, out_dim)
    optimizer = optim.Adam(strategy.parameters(), lr=1e-2)

    for episode in range(300):
        state = env.reset()

        for t in range(200):
            action = strategy.act(state)
            state, reward, done,_ = env.step(action)
            strategy.rewards.append(reward)
            env.render()

            if done:
                break

        loss = train(strategy, optimizer)
        total_reward = sum(strategy.rewards)
        solved = total_reward > 195.0

        strategy.onpolicy_reset()
        print(f'Episode {episode}, loss: {loss}, total reward: {total_reward}, solved: {solved}')


if __name__ == '__main__':
    main()
