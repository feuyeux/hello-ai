# GRPO

#### 1. Introduction

We introduce the Group Relative Policy Optimization (GRPO), a variant reinforcement learning (RL) algorithm of Proximal Policy Optimization (PPO).

GRPO **foregoes** the **critic model**, instead **estimating the baseline** from **group scores**, significantly **reducing training resources**.

We also provide a **unified paradigm** to understand different methods, such as Rejection Sampling Fine-Tuning (RFT) , Direct Preference Optimization (DPO), PPO and GRPO.

Based on such a unified paradigm, we find that all these methods are conceptualized as either direct or simplified RL techniques.

We also conduct extensive experiments, e.g., online v.s. offline training, outcome v.s. process supervision, single-turn v.s. iterative RL and so on, to deeply investigate the essential elements of this paradigm.

At last, we explain why our RL boosts the performance of instruction-tuned models, and further summarize potential directions to achieve more effective RL based on this unified paradigm.

#### 4.1.1. From PPO to GRPO

**Proximal Policy Optimization (PPO)**  is **an actor-critic RL algorithm** that is widely used in the RL fine-tuning stage of LLMs. In particular, it optimizes LLMs by maximizing the following surrogate objective:
$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}, \text{clip} \left( \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right)  A_{t} \right]
$$

where 

- $\pi_{\theta}$ and $\pi_{\theta_{old}}$ are the **current** and **old** <u>**policy models**</u>, 
- and $q, o$  are **questions** and **outputs** sampled from the question dataset and the old policy  $\pi_{\theta_{old}}$, respectively. 
-  $\epsilon$ is a **clipping-related** hyper-parameter introduced in PPO for stabilizing training.  
- $A_t$  is the **advantage**, which is computed by applying **Generalized Advantage Estimation (GAE)**, based on the **rewards**  $\{r_{\ge t}\}$  and a **learned value function**  $V_{\psi}$. 

Thus, in PPO, a **value function** needs to be trained alongside the **policy model** and to mitigate over-optimization of the **reward model**, the standard approach is to add a **per-token KL penalty** from a **reference model** in the reward at each token

$$
r_{t} = r_\phi(q, o_{\le t}) - \beta \log\frac{\pi_{\theta}(o_{t}|q, o_{<t})}{\pi_{ref}(o_{t}|q, o_{<t})},
\label{eq:PPO-reward}
$$

where

-  $r_\phi$  is the **reward model**,
-  $\pi_{ref}$ is the **reference model**, which is usually the **initial SFT model**, 
- and $\beta$  is the **coefficient of the KL penalty**.

As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden. Additionally, during RL training, the value function is treated as a baseline in the calculation of the advantage for variance reduction. While in the LLM context, usually only the last token is assigned a reward score by the reward model, which may complicate the training of a value function that is accurate at each token. To address this, we propose **Group Relative Policy Optimization (GRPO)**, which obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline. More specifically, for each question $q$, GRPO samples **a group of outputs** $\{o_1, o_2, \cdots, o_G\}$  from the old policy  $\pi_{\theta_{old}}$  and then optimizes the policy model by maximizing the following objective:
$$
\mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}  \\
&\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left[ \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}, 1 - \epsilon, 1 + \epsilon \right)  \hat{A}_{i,t} \right] - \beta \mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right]\right\}
$$

where 

- $\epsilon$ and $\beta$ are hyper-parameters, 
- and $\hat{A}_{i,t}$  is the **advantage** calculated based on **relative rewards of the outputs** inside each group only, which will be detailed in the following subsections. 

The group relative way that GRPO leverages to calculate the advantages, aligns well with the comparative nature of rewards models,  as reward models are typically trained on datasets of comparisons between outputs on the same question. Also note that, instead of adding KL penalty in the reward, GRPO regularizes by directly adding the **KL divergence** between the **trained policy** and the **reference policy** to the loss, avoiding complicating the calculation of  $\hat{A}_{i,t}$. And different from the KL penalty term , we estimate the KL divergence with the following unbiased estimator: 
$$
\mathbb{D}_{KL}\left[\pi_{\theta} || \pi_{ref}\right] = \frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1,
$$
which is guaranteed to be positive. 

Iterative Group Relative Policy Optimization

**Input** initial policy model $\pi_{\theta_{\text{init}}}$; reward models $r_\phi$; task prompts $\mathcal{D}$; hyperparameters $\epsilon$, $\beta$, $\mu$
policy model $\pi_\theta \leftarrow \pi_{\theta_{\text{init}}}$
**for** ${iteration = 1, \dots, I}$ **do**
  reference model $\pi_{ref} \leftarrow \pi_{\theta}$
  **for** ${step = 1, \dots, M}$ **do**
    Sample a batch $\mathcal{D}_b$ from $\mathcal{D}$
    Update the old policy model $\pi_{\theta_{old}} \leftarrow \pi_{\theta}$ 
    Sample $G$ outputs $\{o_i\}_{i=1}^G \sim \pi_{\theta_{old}} (\cdot \mid q) $ for each question $q \in \mathcal{D}_b$
    Compute rewards $\{r_i\}_{i=1}^{G}$ for each sampled output $o_i$ by running $r_{\phi}$ 
    Compute $\hat{A}_{i,t}$ for the $t$-th token of $o_i$ through group relative advantage estimation.
    **for** ${GRPO iteration = 1, \dots, \mu}$ **do**
     Update the policy model $\pi_{\theta}$ by maximizing the GRPO objective
  Update $r_\phi$ through continuous training using a replay mechanism.
**Output** $\pi_\theta$


#### 4.1.2. Outcome Supervision RL with GRPO

Formally, for each question $q$,  a group of outputs $\{o_1, o_2, \cdots, o_G\}$  are sampled from the old policy model $\pi_{\theta_{old}}$. A reward model is then used to **score** the outputs, yielding $G$ rewards  $\mathbf{r}=\{r_1, r_2, \cdots, r_G\}$ correspondingly. Subsequently, these rewards are normalized by subtracting the group average and dividing by the group standard deviation. Outcome supervision provides the normalized reward at the end of each output $o_i$  and sets the advantages  $\hat{A}_{i, t}$  of all tokens in the output as the normalized reward, i.e., $\hat{A}_{i, t} = \widetilde{r}_i = \frac{r_i- {\rm mean}(\mathbf{r})}{{\rm std}(\mathbf{r})}$, and then optimizes the policy by maximizing the objective.

#### 4.1.3. Process Supervision RL with GRPO

Outcome supervision only provides a reward at the end of each output, which may not be sufficient and efficient to supervise the policy in complex mathematical tasks. We also explore process supervision, which provides a reward at the end of each reasoning step. Formally, given the question $q$ and $G$  sampled outputs $\{o_1, o_2, \cdots, o_G\}$, a process reward model is used to score each step of the outputs, yielding corresponding rewards: $\mathbf{R} = \{ \{r_1^{index(1)},\cdots,r_1^{index(K_1)}\}, \cdots,  \{r_G^{index(1)},\cdots,r_G^{index(K_G)}\} \}$, where $index(j)$ is the end token index of the $j$-th step, and $K_i$ is the total number of steps in the $i$-th output. We also normalize these rewards with the average and the standard deviation, i.e., $\widetilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - {\rm mean(\mathbf{R})}}{{\rm std(\mathbf{R})}}$.
Subsequently, the process supervision calculates the advantage of each token as the sum of the normalized rewards from the following steps, i.e., $\hat{A}_{i, t} = \sum_{index(j) \ge t} \widetilde{r}_i^{index(j)}$,
and then optimizes the policy by maximizing the objective.

#### 4.1.4. Iterative RL with GRPO

As the reinforcement learning training process progresses, the old reward model may not be sufficient to supervise the current policy model.
Therefore, we also explore the iterative RL with GRPO. In iterative GRPO, we generate new training sets for the reward model based on the sampling results from the policy model and continually train the old reward model using a replay mechanism that incorporates 10% of historical data.
Then, we set the reference model as the policy model, and continually train the policy model with the new reward model.

#### 5.2.1. Towards to a Unified Paradigm

In this section, we provide a unified paradigm to analyze different training methods, such as SFT, RFT, DPO, PPO, GRPO, and further conduct experiments to explore the factors of the unified paradigm. 
Generally, the gradient with respect to the parameter $\theta$ of a training method can be written as:

$$
    \nabla_{\theta}\mathcal{J}_{\textcolor{red}{\mathcal{A}}}(\theta) = \mathbb{E}[\underbrace{(q,o) \sim \textcolor{red}{\mathcal{D}}}_{Data \ Source}]\left( \frac{1}{|o|} \sum_{t=1}^{|o|}  \underbrace{GC_{{\mathcal{A}}}(q, o, t, \textcolor{red}{\pi_{{rf}}})}_{Gradient \ Coefficient}  \nabla_{\theta}\log \pi_{\theta}(o_t | q, o_{<t})\right).
$$

There exist three key components: 
1) Data Source $\mathcal{D}$, which determines the training data;
2) Reward Function $\pi_{{rf}}$, which is the source of the training reward signal;
3) Algorithm $\mathcal{A}$: which processes the training data and the reward signal to the gradient coefficient $GC$ that determines the magnitude of the penalty or reinforcement for the data. We analyze several representative methods based on such a unified paradigm:

- Supervised Fine-tuning (SFT): SFT fine-tunes pretrained model on human selected SFT data.
- Rejection Sampling Fine-tuning (RFT): RFT further fine-tunes the SFT model on the filtered outputs sampled from the SFT model based on SFT questions. RFT filters the outputs based on the correctness of their answers.
- Direct Preference Optimization (DPO): DPO further refines the SFT model by fine-tuning it on augmented outputs sampled from the SFT model, using pair-wise DPO loss.
- Online Rejection Sampling Fine-tuning (Online RFT): Different from RFT, Online RFT initiates the policy model using the SFT model and refines it by fine-tuning with the augmented outputs sampled from the real-time policy model.
- PPO/GRPO: PPO/GRPO initializes the policy model using the SFT model and reinforces it with the outputs sampled from the real-time policy model.

#### A. Appendix

##### 1 Supervised Fine-tuning

The objective of Supervised Fine-tuning is maximizing the following objective:

$$
\mathcal{J}_{SFT}(\theta)=\mathbb{E}[q, o \sim P_{sft}(Q, O)]\left(\frac{1}{|o|}\sum_{t=1}^{|o|} \log \pi_\theta(o_t | q, o_{<t})\right).
$$

The gradient of $\mathcal{J}_{SFT}(\theta)$ is:
$$
\nabla_{\theta}\mathcal{J}_{SFT} = \mathbb{E}[q, o \sim P_{sft}(Q, O)]\left(\frac{1}{|o|}\sum_{t=1}^{|o|} \nabla_{\theta} \log \pi_\theta(o_{t} | q, o_{<t})\right).
$$

Data Source: The dataset employed for SFT. Reward Function: This can be regarded as human selection. Gradient Coefficient: always set to 1.

##### 2 Rejection Sampling Fine-tuning

Rejection Sampling Fine-tuning first samples multiple outputs from the supervised fine-tuned LLMs for each question, and then trains LLMs on the sampled outputs with the correct answer.
Formally, the objective of RFT is to maximize the following objectives:

$$
\mathcal{J}_{RFT}(\theta)= \mathbb{E}[q \sim P_{sft}(Q), o \sim \pi_{sft}(O|q)]\left( \frac{1}{|o|}\sum_{t=1}^{|o|} \mathbb{I}(o) \log \pi_\theta(o_{t} | q, o_{<t})\right).
$$

The gradient of $\mathcal{J}_{RFT}(\theta)$ is:

$$
\nabla_{\theta}\mathcal{J}_{RFT}(\theta)= \mathbb{E}[{q \sim P_{sft}(Q), o \sim \pi_{sft}(O|q)}]\left( \frac{1}{|o|}\sum_{t=1}^{|o|} {\mathbb{I}(o)} \nabla_{\theta}\log \pi_\theta(o_{t} | q, o_{<t})\right).
$$

Data Source: question in SFT dataset with outputs sampled from SFT model. Reward Function: Rule (whether the answer is correct or not). Gradient Coefficient:

$$
GC_{RFT}(q, o, t) = \mathbb{I}(o)=\left\{
\begin{aligned}
1  & & {\rm the \ answer \ of \ o \ is \ correct} \\
0  & & {\rm the \ answer \ of \ o \ is \ incorrect} \\
\end{aligned}
\right.
$$

##### 3 Online Rejection Sampling Fine-tuning

The only difference between RFT and Online RFT is that the outputs of Online RFT are sampled from the real-time policy model $\pi_{\theta}$, rather than from the SFT model $\pi_{\theta_{sft}}$. Therefore, the gradient of online RFT is:

$$
\nabla_{\theta}\mathcal{J}_{OnRFT}(\theta)= \mathbb{E}[{q \sim P_{sft}(Q), o \sim \pi_{\theta}(O|q)}]\left( \frac{1}{|o|}\sum_{t=1}^{|o|} {\mathbb{I}(o)} \nabla_{\theta}\log \pi_\theta(o_{t} | q, o_{<t})\right).
$$

##### 4 Direct Preference Optimization (DPO)

The objective of DPO is:

$$
\begin{split}
    \mathcal{J}_{DPO}(\theta) = \mathbb{E}{[q \sim P_{sft}(Q), o^+, o^- \sim \pi_{sft}(O|q)]} \log \sigma \left(  \beta \frac{1}{|o^+|}\sum_{t=1}^{|o^+|} \log \frac{\pi_{\theta}(o^+_t | q, o^+_{<t})}{\pi_{\text{ref}}(o^+_t | q, o^+_{<t})} - \beta \frac{1}{|o^-|}\sum_{t=1}^{|o^-|} \log \frac{\pi_{\theta}(o^-_{<t} | q, o^-_{<t})}{\pi_{\text{ref}}(o^-_{<t} | q,o^-_{<t})} \right) 
\end{split}
$$

The gradient of $\mathcal{J}_{DPO}(\theta)$ is:

$$
\begin{split}
    \nabla_{\theta}\mathcal{J}_{DPO}(\theta)  = \mathbb{E}{[q \sim P_{sft}(Q), o^+, o^- \sim \pi_{sft}(O|q)]}
     & \left( \frac{1}{|o^+|}\sum_{t=1}^{|o^+|} GC_{DPO}  (q,o,t) \nabla_{\theta}\log\pi_{\theta}(o^+_t | q, o^+_{<t}) \right. \\
    - & \left. \frac{1}{|o^-|}\sum_{t=1}^{|o^-|}  GC_{DPO}  (q,o,t) \nabla_{\theta}\log\pi_{\theta}(o^-_t | q, o^-_{<t}) \right)
\end{split}
$$

Data Source:  question in SFT dataset with outputs sampled from SFT model.
Reward Function: human preference in the general domain (can be `Rule' in mathematical tasks).
Gradient Coefficient: 

$$
GC_{DPO}(q,o,t) = \sigma\left(\beta\log \frac{\pi_{\theta}(o^-_t | q, o^-_{<t})}{\pi_{\text{ref}}(o^-_t | q, o^-_{<t})} - \beta\log \frac{\pi_{\theta}(o^+_t | q, o^+_{<t})}{\pi_{\text{ref}}(o^+_t | q, o^+_{<t})}\right) 
$$


##### 5 Proximal Policy Optimization (PPO)

The objective of PPO is:

$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P_{sft}(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left[ \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}, \text{clip} \left( \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})}, 1 - \epsilon, 1 + \epsilon \right)  A_{t} \right].
$$

To simplify the analysis,  it is assumed that the model only has a single update following each exploration stage, thereby ensuring that $\pi_{\theta_{old}} = \pi_{\theta}$


In this case, we can remove the $\min$ and ${\rm clip}$ operation:

$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P_{sft}(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} \frac{\pi_\theta(o_{t} | q, o_{<t})}{\pi_{\theta_{old}}(o_{t} | q, o_{<t})} A_{t}.
$$

The gradient of $\mathcal{J}_{PPO}(\theta)$ is:

$$
\begin{split}
    \nabla_{\theta}\mathcal{J}_{PPO}(\theta) = \mathbb{E}{[q \sim P_{sft}(Q), o \sim \pi_{\theta_{old}}(O|q)]} \frac{1}{|o|} \sum_{t=1}^{|o|} A_t \nabla_{\theta}\log \pi_\theta(o_{t} | q, o_{<t})
\end{split}
$$

Data Source:  question in SFT dataset with outputs sampled from policy model.

Reward Function: reward model.
Gradient Coefficient:
$$
GC_{PPO}(q, o, t, \pi_{\theta_{rm}}) = A_t,
\label{eq:GC-PPO}
$$
where $A_t$ is the advantage, which is computed by applying Generalized Advantage Estimation (GAE), based on the rewards  $\{r_{\ge t}\}$  and a learned value function  $V_{\psi}$.
$$
%     GC_{PPO}(q, o, t, \pi_{\theta_{rm}}) = A_t(\pi_{\theta_{rm}}) + \beta \left( \frac{\pi_{ref}(o_{t}|o_{<t})}{\pi_{\theta}(o_{t}|o_{<t})}- 1 \right)
% \label{eq:GC-PPO}
% $$
$$
##### 6 Group Relative Policy Optimization (GRPO)}

The objective of GRPO is (assume $\pi_{\theta_{old}} = \pi_{\theta}$ for simplified analysis):
$$
\begin{split}
    \mathcal{J}_{GRPO}(\theta) &= \mathbb{E}{[q \sim P_{sft}(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]}  \\
    & \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[\frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})} \hat{A}_{i,t} - \beta (\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}- \log\frac{\pi_{ref}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})} - 1)\right].
\end{split}
$$
The gradient of $\mathcal{J}_{GRPO}(\theta)$ is:
$$
\begin{split}
    \nabla_{\theta}\mathcal{J}_{GRPO}(\theta)  & = \mathbb{E}{[q \sim P_{sft}(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \\
    & \frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|} \sum_{t=1}^{|o_i|}  
    \left[\hat{A}_{i,t} + \beta \left(\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_{\theta}(o_{i,t}|o_{i,<t})} - 1\right)\right]  \nabla_{\theta}\log \pi_\theta(o_{i,t} | q, o_{i,<t}). 
\end{split}
$$
Data Source:  question in SFT dataset with outputs sampled from policy model.
Reward Function: reward model.
Gradient Coefficient:
$$
    GC_{GRPO}(q, o, t, \pi_{\theta_{rm}}) = \hat{A}_{i,t} + \beta \left(\frac{\pi_{ref}(o_{i,t}|o_{i,<t})}{\pi_{\theta}(o_{i,t}|o_{i,<t})} - 1\right),
\label{eq:GC-GRPO}
$$
where $\hat{A}_{i,t}$ is computed based on the group reward scores.