# 深度学习入门4

<https://github.com/oreilly-japan/deep-learning-from-scratch-4>

```sh
python -m venv rl4_env

source rl4_env/bin/activate
pip install --upgrade pip
pip install matplotlib
# 7
pip install dezero
pip install dezerogym
# 9
pip install gym
```

Bandit

1.1 bandit_avg.py
1.2 non_stationary.py

DP

4.1 dp_inplace.py
4.2 policy_eval.py
4.3 policy_iter.py
4.4 value_iter.py

MC

5.1 mc_eval.py
5.2 mc_control.py
5.3 importance_sampling.py

TD

6.1 td_eval.py
6.2 sarsa.py
6.3 sarsa_off_policy.py
6.4 q_learning.py
6.5 q_learning_simple.py

DRL(Deep Reinforcement Learning)

7.1 dezero1.py
7.2 dezero2.py
7.3 dezero3.py
7.4 dezero4.py
7.5  q_learning_nn.py

8.1 replay_buffer.py 经验回放
8.2 dqn.py 目标网络

9.1 simple_pg.py
9.2 reinforce.py
9.3 actor_critic.py
