# 大语言模型RLHF全链路揭秘:从策略梯度、PPO、GAE到DPO的实战指南

**原创张逸骅 PaperWeekly 2025年02月26日 13:02北京**

©PaperWeekly原创·作者|张逸骅  
单位|密歇根州立大学博士生  
研究方向|可信人工智能

如果你对大语言模型(LLM)的强化学习(RLHF)感兴趣，又想从最基础的策略梯度优化一路了解、推导出PPO、GAE，再深入探讨DPO，那你就来对地方了。

本文将从最基础的 Gradient Policy Optimization开始，逐步介绍经典的 REINFORCE算法，再讲解如何利用剪切目标实现近端策略优化(PPO)，并通过广义优势估计(GAE)在偏差与方差之间找到最佳平衡。之后，我们还会从头推导、讨论离线训练方法，如DPO，帮助你了解不同训练路线的优势与挑战。

## 1. 在线(On-Policy)和离线(Off-Policy)强化学习

如今，LLM中主流的RLHF方向分为两大路线:

- 以PPO为代表的On-Policy路线  
- 以DPO为代表的 Off-Policy路线  

那么，什么是On-Policy，什么是Off-Policy呢?可以用一个简洁的判定方法:

- On-Policy:训练过程中，需要模型亲自参与"生成"来收集新的数据样本。  
- Off-Policy:训练过程中，不需要"在线"生成，更多依赖事先收集到的(或由别的策略产生的)数据进行离线学习。

一般来说，On-Policy的方法在训练时会更"耗卡"、更耗时一一最大的开销主要是源自"模型生成"这一步，因为对一个生成式任务而言，模型需要逐token地输出，这个过程极其耗费算力。

不过，尽管速度较慢，On-Policy在理论上拥有更高的效果上限，因为它能够不断根据当前模型状态进行探索和更新，这一点将在后续讨论PPO时更加凸显。

我们首先来谈谈On-Policy路线。On-Policy的核心思路是:让模型自己产出答案，然后依据答案的优劣来打分，以此指导下一步的参数更新。简而言之，最关键的一点是让模型"亲自下场"。

假设你是一个需要学会下象棋的模型，现在有两种训练方式:

1. 方式一:让你真刀真枪地下棋，每一步都有教练跟在你身边打分。当你吃掉对手棋子时，教练会鼓励你;当你因为冲动失误被对面反杀时，教练会及时提醒你改进。
2. 方式二:给你一堆职业选手的比赛录像和一堆臭棋篓子的对局，用标注告诉你哪些操作是好招，哪些操作是坏招，然后你被动地学这些好操作、避免坏操作。

这两种方式最大的区别就在于:你有没有亲自去"下棋"。方式一就是On-Policy，需要模型自己产出行为，然后学习;方式二就是Off-Policy，只需根据已有对局数据进行模仿式学习。

Off-Policy在训练时通常更快，因为它用现成的数据就可以了，不需要模型时时在线生成并等待打分，但也很依赖这批数据与当前模型能力的"匹配度"。如果数据中操作难度和模型水平相差太大(过高或过低)，学习效果就可能大打折扣;On-Policy则可以避免这一问题，因为它所得到的训练样本100%来自于自己当前的水平和行动。

在语言模型场景中，一个典型的On-Policy算法往往包含以下组件:

- Actor:负责"生成"句子的模型(就像正在对弈的你)。  
- Critic:类似于"教练"，为每个生成结果提供即时指导;它本身也在训练过程中随Actor的能力变化而调整。  
- Reward Model:相当于"裁判"，给出最终分数或偏好评估。通常在训练过程中是固定不动的。  
- Reference Model:PPO在大模型里的"独有角色"，用来防止Actor过度偏离原有预训练分布，缓解reward hacking等问题。  

由于在大型LLM上，这四个部分的参数量都可能非常庞大(往往需要同时加载多个70B参数规模的模型)，所以On-Policy训练往往带来极高的算力需求，这也是为什么人们通常说PPO"非常耗卡"的原因。

下一步，我们将把目光聚焦在当前On-Policy路线最具代表性的方法--PPO上，看看它究竟如何在实践中平衡训练开销与学习效率。

## 2. PPO(近端策略优化)

### 2.1 从策略梯度优化(Policy Gradient Optimization)谈起

想象一下，你是一名刚开始学习下象棋的新手。你的目标是通过不断调整你的下棋策略(记作 $\pi_{\theta}$ ，其中 $\theta$ 表示你的策略参数)，来提高在一局棋中获得胜利的概率，也就是最大化你的期望回报。我们可以将每一盘棋看作是一条轨迹 $\tau$ ，而你要做的，就是通过不断优化你的策略来获得更高的回报。

更一般得，强化学习的目标就是去优化一个策略，使得回报的期望最大:

$$\pi^{*}=\arg\max_{\pi}J(\pi)$$

形式上，这个策略的回报被定义在所有可能的轨迹上:

$$J\left(\pi_{\theta}\right)=\int_{\tau} P(\tau\mid\pi) R(\tau)=E_{\tau\sim\pi}[R(\tau)]$$

所谓的轨迹，就是一连串状态和对应动作的组合(state,action):

$$\tau=\left(s_{0}, a_{0}, s_{1}, a_{1},\ldots\right)$$

在下棋这个例子中，状态 $s_{t}$ 可以理解为当前棋盘落子的状态，而动作 $a_{t}$ 即为下一次落子的地方。而当前时间点的下一个状态，则服从某种概率分布，可以被看作是随机的、不确定的(即对手落子):

$$s_{t+1}\sim P\left(\cdot\mid s_{t}, a_{t}\right)$$

那么一个轨迹 $\tau$ 的概率则为:

$$P(\tau\mid\pi)=\rho_{0}\left(s_{0}\right)\prod_{t=0}^{T-1} P\left(s_{t+1}\mid s_{t}, a_{t}\right)\pi\left(a_{t}\mid s_{t}\right)$$

在强化学习中，我们会不断提到回报会随着时间不断打折(discount reward)的概念:未来的回报总是不如当下的回报那么重要。所以一个策略 $\tau$ 的总回报可以被视作:

$$R(\tau)=\sum_{t=0}^{\infty}\gamma^{t} r_{t}$$

其中 $\gamma\in[0,1]$ 是时间上的折扣因子，而 $r_{t}$ 是t时刻的实际回报。

在深度学习中，我们通常采用最小化损失函数来更新参数，这正是随机梯度下降(Stochastic Gradient Descent)的做法。但在这里，我们的目标是最大化回报，因此我们使用随机梯度上升(Stochastic Gradient Ascent)来更新策略:

$$\theta_{k+1}=\theta_{k}+\alpha\nabla_{\theta} J\left(\pi_{\theta}\right)\left|_{\theta_{k}}\right.$$

这里的 $\nabla_{\theta}J(\pi_{\theta})$ 就被称为策略梯度(policy gradient)。换句话说，就好比每盘棋结束后，你会复盘，评估自己每一步走法对最终胜负的贡献，然后调整下一盘棋的策略。这样的更新方法统称为策略梯度算法(policy gradient algorithms)。

然而，正如在下棋时要考虑所有可能的走法和局面一样，精确计算这个梯度需要对所有可能棋局(轨迹)进行求和或积分，而这在实际中(除非棋盘极其简单)是计算上不可行的，因为即使你拥有可导的 $R(\tau)$ ，由于轨迹的步数太多，在使用 auto-differentiation求导过程中会因为memory太大而使用非常受限。因此，我们需要仔细思考一下怎么求这个策略梯度。

### 策略梯度的推导

为了得到一个可操作的策略梯度公式，就像在复盘中总结经验一样，我们从目标函数的梯度开始推导。将每盘棋视为一条轨迹 $\tau$ ,目标函数梯度为:

$$\nabla_\theta J\left(\pi_\theta\right)=\nabla_\theta E_{\tau\sim\pi_\theta}[R(\tau)]$$

#### 第1步:展开期望(Expand the Expectation)

这一步相当于考虑所有可能的棋局，我们将期望展开为对所有轨迹的积分:

$$=\nabla_\theta\int_\tau P(\tau\mid\theta) R(\tau) d\tau$$

#### 第2步:交换梯度与积分(Interchange Gradient and Integral)

就像把每一步棋的影响拆分出来,我们将梯度操作符移入积分内部:

$$=\int_{\tau}\nabla_\theta P(\tau\mid\theta) R(\tau) d\tau$$

#### 第3步:使用对数导数技巧(Apply Log-Derivative Trick)

利用一个数学技巧(对数导数),类似于在复盘中分解每一步的重要性,我们有:

$$=\int_{\tau} P(\tau\mid\theta)\nabla_\theta\log P(\tau\mid\theta)\cdot R(\tau)\, d\tau$$

#### 第4步:回到期望形式(Return to Expectation Form)

最终，我们可以把上面的积分重新写成期望的形式:

$$=E_{\tau\sim\pi_{\theta}}\left[\nabla_{\theta}\log P(\tau\mid\theta)\cdot R(\tau)\right]$$

### 分解 $\nabla_{\theta}\log P(\tau\mid\theta)$ 

在下棋的过程中，每盘棋的走法取决于你每一步的决策。假设一盘棋的轨迹 $\tau$ 可表示为:

$$P(\tau\mid\theta)=\rho_{0}\left(s_{0}\right)\prod_{i=0}^{T-1} P\left(s_{i+1}\mid s_{i}, a_{i}\right)\pi_{\theta}\left(a_{i}\mid s_{i}\right)$$

这里 $\pi_{\theta}\left(a_{i}\mid s_{i}\right)$ 就是你在棋局某一时刻(状态 $s_{i}$ )下选择某一步棋(动作 $a_{i}$ )的概率。取对数后求梯度，我们得到:

$$\nabla_\theta\log P(\tau\mid\theta)=\sum_{i=0}^T\nabla_\theta\log\pi_\theta\left(a_i\mid s_i\right)$$

(注意:棋局中对手的反应 $P\left(s_{i+1}\mid s_{i},a_{i}\right)$ 由规则决定，与 $\theta$ 无关，因此其梯度为零。也就是说，当 $s_{i},a_{i}$ 给定时， $s_{i+1}$ 也就定下来了。)

### 2.2 最终策略梯度公式(Final Policy Gradient Formula)

把上面的结果代入期望，我们最终得到的公式是:

$$\nabla_\theta J(\pi_\theta)=E_{\tau\sim\pi_\theta}\left[\sum_{i=0}^T\nabla_\theta\log\pi_\theta(a_i\mid s_i)\cdot R(\tau)\right]$$

在这条公式中，每一步棋的决策 $\left(\log\pi_{\theta}\right)$ 决定了整盘棋的表现，而不依赖于对手的固定规则。实际操作中，我们通常使用蒙特卡洛抽样来近似这个期望，就好比你通过大量实战积累经验来提升下棋水平。最后，基于采样的策略梯度可以由以下式子近似:

$$\hat{g}=\frac{1}{\mathcal{D}}\sum_{\tau\in\mathcal{D}}\sum_{t=0}^{T}\nabla_\theta{ log}\pi_\theta\left(a_t\mid s_t\right) R(\tau)$$

如果你仔细观察这个式子，你会发现很有意思的两个地方。首先 $R(\tau)$ 直接出现在策略参数的梯度里边。

### 2.3 REINFORCE算法流程与实现步骤

下面介绍经典的策略梯度方法一一REINFORCE算法，它就像你通过不断下棋、复盘总结经验来不断改进你的棋艺:

#### 1.策略网络构建

搭建一个神经网络来定义你的下棋策略 $\pi_{\theta}$ :

- 输入:当前棋局状态 $s_{t}$  
- 输出:根据当前棋局生成下一步棋的概率分布 $P(a_{t}\mid s_{t})$  

#### 2.轨迹采样

用当前策略进行对局(采样轨迹 $\tau$ )，并记录每步棋得到的奖励(例如赢棋后的奖励分数)。

- 你可以设定每盘棋固定步数(比如100步)，或直到比赛结束。

#### 3.梯度计算

根据收集到的对局数据集 $\mathcal{D}$ 计算梯度估计，就像总结每盘棋中各步对胜负的贡献:

$$\hat{g}=\frac{1}{|\mathcal{D}|}\sum_{\tau\in\mathcal{D}}\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}\left(a_{t}\mid s_{t}\right) R(\tau)$$

#### 4.参数更新

使用随机梯度上升法更新你的策略参数，就好像根据复盘结果调整你的下棋风格:

$$\theta_{k+1}=\theta_{k}+\alpha\hat{g}$$

或者写作:

$$\theta_{k+1}=\theta_{k}+\left.\alpha\nabla_{\theta} J\left(\pi_{\theta}\right)\right|_{\theta_{k}}$$

#### 5.循环优化

重复"下棋-复盘-调整"这一过程，直到你的策略收敛，即你能稳定地打出高水平的棋局。

### 核心公式说明

#### 1.梯度估计公式

$$\hat{g}=\frac{1}{|\mathcal{D}|}\sum_{\tau\in\mathcal{D}}\sum_{t=0}^T\underbrace{\nabla_\theta\log\pi_\theta\left(a_t\mid s_t\right)}_{\text{每步棋的决策梯度}}\cdot\underbrace{R(\tau)}_{\text{整盘棋的总奖励}}$$

- 这里，我们利用大量实战(蒙特卡洛抽样)来近似整个期望。  
- 轨迹总奖励 $R(\tau)$ 就是整盘棋的胜负结果,用来衡量你所有决策的综合效果。  

#### 2.参数更新规则

$$\theta_{k+1}=\theta_{k}+\alpha\hat{g}$$

- $\alpha$ 表示学习率,相当于每盘棋复盘后你调整策略的幅度;梯度的方向正是指向能够提升胜率的方向。  

#### 3.算法特性

- 关键优势:这种方法完全依靠你下棋的实战经验,不需要提前知道对手的策略(model-free)。  
- 计算要求:需要大量的对局采样以降低随机性带来的波动(方差)。  
- 改进方向:后续方法(如 Actor-Critic)会引入价值函数参考线，使得策略更新更为稳定，就像在复盘中加入专业教练的点评一样，帮助你更快提高棋艺。  

### 2.4 策略梯度优化面临的问题

策略梯度优化一个核心的假设是:我们可以通过采用的方法来估计策略的梯度。但是当问题的规模变得非常大:比如每次轨迹 $\tau$ 都非常长，又或者策略模型非常大，为了预估准确的梯度，我们就不得不采样多次，否则就会面临方差很高的问题。

策略梯度算法中的梯度估计虽然在理论上是无偏的(即其期望值会收敛到真实梯度)，但实际上它的方差非常高，该梯度估计可以写成:

$$\hat{g}=\frac{1}{|\mathcal{D}|}\sum_{\tau\in\mathcal{D}}\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t\mid s_t) R(\tau)$$

其中:

- $|\mathcal{D}|$ 表示数据集 $\mathcal{D}$ 的大小，  
- $\pi_{\theta}$ 是当前策略(你的下棋策略)，  
- $R(\tau)$ 是整盘棋(轨迹 $\tau$ )的总回报,  
- $a_{t},s_{t}$ 分别代表在时间步 t你采取的动作和所处的状态。想象你在下棋。

想象你在下棋。每一步你都希望知道自己的决策对最终胜负的贡献，但问题在于，如果你试图把整盘棋的输赢都归因于每一步决策，那么这种评估就会变得非常不稳定--也就是方差很
高。接下来，我们将采取不同的做法来减小这样估计的方差。

### 2.5 减小方差:只关注未来

观察上边用于梯度估计得式子:无论当前在哪一步t， $R(\tau)$ 总是会把整个轨迹中所有的reward都算进去。然后这么做是不太合理的，当前的决策应该只需要考虑对未来产生的影响:过去的已经无法改变了，无需再加入到 $R(\tau)$ 的计算中。

回到下棋的例子:假如每一步的评分都把前面已经走过的好步或坏步也计入进去，那就会混淆你当前决策的真实价值。实际上，在评估当前走法时，你只需要关注从这一步开始直到局末的"后续表现"。这就是所谓的"rewards to go"，即只考虑从当前动作开始到比赛结束所获得的奖励。

用数学表达就是，我们将原来的梯度估计调整为:

$$\begin{align*}\nabla_\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N\left(\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\right)\left(\sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})\right)\\ \end{align*}$$

这里， $\sum_{t^{\prime}=t}^{T}r(s_{i,t^{\prime}},a_{i,t^{\prime}})$ 就代表了从当前走法开始到局末的"奖励总和"。这样做就像你在复盘时，只关注从某一步开始后续的变化，而不纠结于那一步之前已经发生的事情。

因此，当我们把这些"来自过去的"的冗余项去除后，噪声就自然而然会减少一些。

### 2.6 减小方差:参考线(Baseline)

为了进一步减少评估中的波动，我们可以为每一步的"后续奖励"减去一个基准值。数学上，这个参考线通常记为b(在实际中，我们常用价值函数 $V^{\pi}(s)$ 来作为这个参考线)，公式为:

$$\begin{align*}\nabla_\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N\left(\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\right)\left(\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})- b\right)\end{align*}$$

其中b就是所谓的参考线，他不一定是一个常数，更多时候是另外一个状态 $s_{t}$ 的函数。这个参考显得实际意义是，在当前的状态下，回报的期望大概是什么样子。那么所谓超出期望的部分 $\sum_{t^{\prime}=t}^{T}r(s_{i,t^{\prime}},a_{i,t^{\prime}})-b$ 就是优势(Advantage)。在实际训练中，我们会用优势代替原来的奖励进行梯度估计，以减小方差。

在大语言模型的对齐训练中，我们通常在语言模型(即策略 $\pi_{\theta}$ )的基础上增加一个额外的线性层，用来估计在某个状态下的预期回报 $V^{\pi}(s)$ 。这相当于为每个局面设定了一个标准分，帮助我们衡量当前决策的实际优势。如果你想更直观得了解为什么需要这个参考线，可以阅读我的上一篇文章。

### 2.7 减小方差:引入Q和V

在上边，我们提到了"rewards to go"的概念，即 $\sum_{t^{\prime}=t}^{T}r(s_{i,t^{\prime}},a_{i,t^{\prime}})$ 。这个项在强化学习中被称为Q函数( $Q^{\pi}(s,a)$ )，即在状态s采取动作a后，未来获得的总回报。然后，通过减去状态价值 $V^{\pi}(s)$ 我们得到优势函数:

$$A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$$

用下棋的比喻，Q函数描述的是你当前这个局面 s在走了一步 a后可能的输赢，而状态价值表示的是仅凭现在的局面，你还有多少老本儿可以吃。如果当前局面对你十分有利，但是你走了一步臭棋，尽管你最后赢面还是很大(相当于 $Q^{\pi}(s,a)$ 的绝对大小还是很大)，但是你相对于对方的"优势"却减弱了。

所以，你不仅应该关注某一步棋的绝对得分，还要和"老本儿"比较比较，看看这一步棋究竟为你增加了多少胜率。如果 $A^{\pi}(s,a)$ 为正，则说明这步棋明显扩大了你的优势;若为负，则表明你这一招不妙。

最终，我们可以将策略梯度写为:

$$\begin{align*}\nabla_\theta J(\theta)&\approx\frac{1}{N}\sum_{i=1}^N\left(\sum_{t=0}^T\nabla_\theta\log\pi_\theta(a_{i,t}\mid s_{i,t})\right) A^\pi(s,a)\end{align*}$$

这公式正是衡量你在每个局面上，通过比对每一步棋与平均表现的差距，从而决定如何调整你的下棋策略。

### 2.8 优势函数的解释

简单来说，优势函数 $A^{\pi}(s,a)$ 就告诉你，在某个局面(状态s)下，选择某个特定走法(动作a)相比于平均走法能提升多少胜率。如果这步棋带来的预期回报远高于你当前的基准水平，那么这步棋的优势就是正的，说明它非常值得采用;反之，则说明不如平均水平。

总之，通过这些方法一一只考虑"后续奖励"、引入参考线以及使用优势函数，我们就能在训练中有效降低梯度估计的方差，就像你在下棋时只关注关键走法对局面转变的影响，从而让策略更新更稳定、更有针对性。

### 2.9 如何估计优势项-使用基于迭代的 GAE策略

我们可以用多种方式来估计优势项。例如:

$$\begin{gathered}
\hat{A}^\pi(s_t, a_t)=\left[r\left(s_t, a_t\right)+\gamma V^\pi(s_{t+1})\right]- V^\pi(s_t)\\
\hat{A}^\pi(s_t, a_t)=\left[r\left(s_t, a_t\right)+\gamma r\left(s_{t+1}, a_{t+1}\right)+\gamma^2 V^\pi(s_{t+2})\right]- V^\pi(s_t)\\
\hat{A}^\pi(s_t, a_t)=\left[r\left(s_t, a_t\right)+\gamma r\left(s_{t+1}, a_{t+1}\right)+\gamma^2 r\left(s_{t+2}, a_{t+2}\right)+\gamma^3 V^\pi(s_{t+3})\right]- V^\pi(s_t)
\end{gathered}$$

上边的例子告诉我们，我们可以累加若干步来实现偏差和方差的权衡。

- 如果我们过早地停止累加真实的奖励项:就会产生高偏差(high bias)，因为只使用了对价值函数的小部分近似和极少的真实奖励。  
-  如果我们累加过多的奖励项:则会引入高方差(high variance)，因为依赖更多真实采样会让估计量不稳定。  

为平衡这一偏差-方差问题，我们可以采用对这几项进行加权求和的做法，也就是广义优势估计(Generalized Advantage Estimation, GAE):

$$\begin{array}{c}
\delta_{t}=r_{t}+\gamma V^{\pi}\left(s_{t+1}\right)-V^{\pi}\left(s_{t}\right)\\
\hat{A}_{t}=\delta_{t}+\gamma\lambda\hat{A}_{t+1}
\end{array}$$

- 这是一个递归公式。末端时刻的优势估计可以看作第一种展开，而它的前一时刻会再加上一层衰减系数λ。  
- 通过在各时间步上不断迭代累加，就可以平衡真实奖励所带来的高方差和使用价值函数所带来的高偏差。

下一章会讲述 GAE的详细推导。

在大语言模型对齐的场景中，这个结果会指示策略(语言模型)在给定某个提示(state)后，去提升那些在期望意义上"优于平均"奖励的下一个token选取概率。换言之，模型将倾向于选择那些更可能引导未来token合乎我们所希望的奖励标准(即更"对齐"或更符合训练数据分布)之序列。

### 2.10 PPO损失函数(The PPO Loss)

在PPO(近端策略优化)中，为了防止更新时策略变化过大，我们会构造一套特殊的损失函数。它主要由以下几部分构成:

**策略损失(Policy Loss, $L_{POLICY}$)**

$$\begin{align*} 
L_{POLICY}&=\min\left(\frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}\hat{A}_t,\,clip\left(\frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)},\, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\\ 
\end{align*}$$

这一部分就像是在下棋时，你不希望一次改变策略太多，而是希望微调每一步的选择，保证既能改善局势，又不会因冒险走出常规而导致局面混乱。

**价值函数损失(Value Function Loss, $L_{V F}$)**

$$ L_{VF}=\frac{1}{2}\left\|V_\theta(s)-\left(\sum_{t=0}^T\gamma^t r_t\mid s_0=s\right)\right\|_{2}^2$$

这一项帮助我们确保对于每个局面，你的预期回报估计(就像预判棋局发展)与实际获得的回报尽可能接近。

**熵损失(Entropy Loss, $L_{ENTROPY}$)**

$$L_{ENTROPY}=-\sum_{x} p(x)\log p(x)$$

熵损失鼓励策略保持一定的探索性，就像一个优秀的棋手不仅熟练掌握定式，同时也敢于尝试新变化，保持灵活应变的能力。

**PPO总损失(PPO Loss, $L_{PPO}$)**

$$L_{PPO}=L_{POLICY}+c_{1} L_{VF}+c_{2} L_{ENTROPY}$$

将这些部分结合起来，就构成了PPO的总损失函数。这个损失函数旨在在更新策略时既提高胜率(奖励)，又防止策略偏离原有风格过远，保持平稳而高效的改进。

### 2.11 使用PPO的优势

- **稳定性**:剪切操作(clipping)确保策略更新时步伐不会过大，就像你在下棋时不会突然改变风格，保证每一步都稳扎稳打。  
- **样本效率**:PPO能够较好地利用收集到的对局数据，尽管在大型模型的场景下仍需大量采样。  
- **内在安全性**:通过剪切更新和参考策略的 KL惩罚，PPO能有效防止模型在更新时出现剧烈偏差，从而确保生成结果不会与预训练风格南辕北辙。  

总体来说，就像一位经验丰富的棋手在不断下棋、复盘、调整策略中不断进步一样，PPO通过精细的梯度更新和对策略变化的限制，实现了稳健而高效的强化学习。

## 3. GAE(广义优势估计)理解及推导

### 3.1 GAE公式推导

广义优势估计(Generalized Advantage Estimation, GAE)通过引入衰减系数λ来平衡TD误差的多步估计。首先定义单步TD误差：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GAE将k步TD误差进行指数加权平均：

$$\hat{A}_t^{GAE} = \delta_t + \gamma\lambda\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots + (\gamma\lambda)^{T-t-1}\delta_{T-1}$$

这可以展开为递归形式：

$$\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$$

### 3.2 解析解推导

将递归式展开得到闭式解：

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$$

代入δ的定义：

$$= \sum_{l=0}^{\infty}(\gamma\lambda)^l[r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})]$$

通过望远镜求和(telescoping sum)可得：

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty}(\gamma\lambda)^l r_{t+l} - V(s_t)$$

### 3.3 偏差-方差权衡

当λ=0时：  
$$\hat{A}_t^{GAE} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$  
→ 高偏差，低方差  

当λ=1时：  
$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty}\gamma^l r_{t+l} - V(s_t)$$  
→ 无偏差，高方差  

通过调节λ∈[0,1]，可以在偏差与方差之间取得最佳平衡。

## 4. PPO的完整算法流程

### 4.1 算法步骤
1. **初始化**：创建策略网络πθ，价值网络Vφ
2. **数据收集**：
   - 用当前策略πθ生成轨迹{τ}
   - 记录(st, at, rt, st+1)元组
3. **优势计算**：
   - 用GAE计算每个时间步的$\hat{A}_t$
4. **优化迭代**：
   - 对minibatch数据计算：
     - 策略损失$L^{CLIP}$
     - 价值损失$L^{VF}$
     - 熵奖励$L^{ENT}$
   - 梯度下降更新θ, φ
5. **重复**直到收敛

### 4.2 关键技术细节
1. **梯度裁剪**：  
   ```python
   ratio = prob_ratio = πθ(a|s)/π_old(a|s)
   clipped_ratio = torch.clamp(ratio, 1-ε, 1+ε)
   loss = -torch.min(ratio * A, clipped_ratio * A).mean()
   ```

2. **价值网络更新**：  
   $$L^{VF} = \frac{1}{2}\|V_\phi(s_t) - V_{target}\|^2$$  
   $$V_{target} = \hat{A}_t + V_{\phi_{old}}(s_t)$$

3. **自适应KL惩罚**：  
   $$L^{KL} = \beta \cdot KL[\pi_{old} \| \pi_\theta]$$  
   动态调整β保持KL散度在目标区间

## 5. DPO(直接偏好优化)

### 5.1 动机与原理
针对PPO的缺陷提出离线优化方法：
- 不需要在线生成
- 直接利用偏好数据(y_w ≻ y_l | x)
- 推导基于Bradley-Terry模型的损失函数：

$$\mathcal{L}_{DPO} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

### 5.2 完整推导过程

1. **奖励建模**：  
   $$r(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

2. **偏好概率**：  
   $$P(y_w \succ y_l|x) = \frac{\exp(r(x,y_w))}{\exp(r(x,y_w)) + \exp(r(x,y_l))}$$

3. **极大似然估计**：  
   $$\max_\theta \mathbb{E}_{(x,y_w,y_l)}[\log P(y_w \succ y_l|x)]$$

4. **最终损失函数**：  
   $$\mathcal{L}_{DPO} = -\log\sigma\left(r(x,y_w) - r(x,y_l)\right)$$

### 5.3 实现关键点
1. **参考模型冻结**：π_ref参数固定
2. **双样本处理**：同时处理(y_w, y_l)对
3. **温度系数β**：控制策略偏离参考模型的程度
4. **数据增强**：对每个prompt x生成多个响应

## 6. 对比分析与应用实践

### 6.1 PPO vs DPO
| 维度               | PPO                          | DPO                    |
|--------------------|------------------------------|------------------------|
| 数据需求           | 需要在线生成                | 依赖静态偏好数据集    |
| 计算成本           | 高(需多个模型同时加载)      | 较低(单模型+参考模型) |
| 训练稳定性         | 需要精细调参                | 相对稳定              |
| 策略探索能力       | 强(在线探索)                | 有限(受限于数据集)    |
| 实际效果上限       | 理论更高                    | 依赖数据质量          |
| 典型应用场景       | 对话系统、复杂任务          | 风格迁移、简单对齐    |

### 6.2 实践建议
1. **硬件配置**：
   - PPO需要至少4×80G A100完整加载：
     - Actor(33B)
     - Critic(33B)
     - Reward Model(33B)
     - Reference Model(33B)
   - DPO只需2×80G A100：
     - Trainable Model(33B)
     - Reference Model(33B)

2. **调参经验**：
   - PPO关键参数：
     ```yaml
     clip_range: 0.2 
     gamma: 0.99
     gae_lambda: 0.95
     vf_coef: 0.5 
     ent_coef: 0.01
     ```
   - DPO关键参数：
     ```yaml
     beta: 0.1-0.5
     learning_rate: 1e-6
     per_device_batch_size: 16
     ```

3. **收敛判断**：
   - PPO：监控KL散度(建议0.5-2 nats)
   - DPO：跟踪偏好准确率(>75%说明有效)

## 7. 前沿方向展望

### 7.1 混合训练框架
- **PPO+DPO联合训练**：初期用DPO快速对齐，后期用PPO精细优化
- **课程学习设计**：逐步增加生成难度
- **多目标优化**：同时优化多个奖励信号

### 7.2 理论突破方向
1. **分布偏移理论**：形式化证明offline RL的收敛边界
2. **最优传输应用**：提升策略更新的几何特性
3. **因果推理整合**：建立生成token的因果归因模型

### 7.3 工程优化趋势
1. **模型架构**：
   - LoRA适配器降低显存消耗
   - 共享底层参数(如Critic复用Actor的transformer层)
2. **系统优化**：
   - 流水线并行生成
   - 异步奖励计算
3. **数据工程**：
   - 自动生成偏好对
   - 噪声标签清洗

> 本文完整代码实现已开源在：https://github.com/RLHF-FullStack/PPO-DPO-Tutorial
```