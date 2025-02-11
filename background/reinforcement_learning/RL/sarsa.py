# Description: SARSA (State-Action-Reward-State-Action) algorithm implementation
from common.utils import greedy_probs
from common.gridworld import GridWorld
import numpy as np
from collections import defaultdict, deque
import os
import sys
# for importing the parent dirs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 同策略型


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        # 使用deque存储两个连续的状态
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        # 从策略中选择一个动作
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        # SARSA算法更新Q值
        next_q = 0 if done else self.Q[next_state, next_action]

        # 使用TD方法进行更新
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 策略的改进
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = SarsaAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)

        if done:
            # 到达目标时也要调用
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q)
