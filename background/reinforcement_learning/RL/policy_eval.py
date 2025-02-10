# Description: 策略评估

if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
from common.gridworld import GridWorld


def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        # 目的地的价值函数总是为0
        if state == env.goal_state:
            V[state] = 0
            continue
        # probabilities of each action
        action_probs = pi[state]
        new_V = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 新的价值函数
            new_V += action_prob * (r + gamma * V[next_state])
        V[state] = new_V
    return V


def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        # 更新前的价值函数
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        # 求更新量的最大值
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        # threshold: 进行策略评估时，停止更新的阈值
        if delta < threshold:
            break
    return V


if __name__ == '__main__':
    env = GridWorld()
    # 折现率
    gamma = 0.9
    # 策略
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    # 价值函数
    V = defaultdict(lambda: 0)

    V = policy_eval(pi, V, env, gamma)
    # 环境
    env.render_v(V, pi)
