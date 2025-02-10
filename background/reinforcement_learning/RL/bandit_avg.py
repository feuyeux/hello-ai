import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms=10):
        # 稳态问题(stationary problem)：老虎机的价值是固定的，不会随时间变化
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        # Qs的每个元素存储的是对应的老虎机价值的估计值
        self.Qs = np.zeros(action_size)
        # ns的每个元素存储的是对应的老虎机被玩过的次数
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        # ε-greedy算法
        if np.random.rand() < self.epsilon:
            # 探索(exploration) 为了对老虎机的价值做出更准确的估计，尝试不同的老虎机
            return np.random.randint(0, len(self.Qs))
        # 利用(exploitation) 利用目前实际的游戏结果，玩那些看起来最好的老虎机(贪婪行动)
        return np.argmax(self.Qs)
    
runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))  # (2000, 1000)

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))

    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()
