+++
title = "Why does experience replay require off-policy learning and how is it different from on-policy learning?"
date = 2024-05-04T11:39:36-04:00
draft = false
+++

When you use an experience replay buffer, you save the most recent $k$ experiences of the agent, and sample data from that buffer for training. Typically, the agent does a step of training to update its policy for every step in the environment. At any moment in time, the vast majority of experiences in the buffer are generated with a different -- earlier -- policy than the current policy. And if the policy used to collect data is different than the policy being evaluated or improved, then you need an off-policy method.<!--more-->

## Off-policy vs on-policy

There is often confusion about the meaning of off-policy and on-policy. Many people believe "on-policy" refers to any method that evaluates an explicit policy. These people would consider any actor-critic method to be an "on-policy" method because in actor-critic methods, the actor is an explicit parameterized policy that the critic (the value function) evaluates. While many actor-criticm methods are on-policy, the term classically means something different, and you can in fact have off-policy actor-critic methods.

The classic meaning of on-policy vs off-policy regards whether your training method requires your training data to be collected from the policy to be evaluated and improved, or whether it can be used even if your data is collected from a different policy.

If you hold your policy constant and then collect a buch of data with it, then this data distribution is on-policy and you can use an on-policy method to evaluate/improve it.

If your data is generated by some other policy, be it an exploration policy, older versions of your policy, or maybe even some other expert, then you will need an off-policy method. Since the experience replay buffer is dominated by data generated by earlier versions of the agent's policy, you will need an off-policy method to do policy evaluation/improvement from it.

## Evaluation vs improvement and the strange case of PPO

You may have noticed I keep naming two cases where on/off policy is relevant: for policy evaluation and policy improvement. For most algorithms, both the evaluation and improvement will be the same: either on-policy or off-policy. However, evaluation and improvement are two distinct steps. You could have one part be on-policy while the other is off-policy. PPO is an example where the policy evaluation is on-policy while the improvment is off-policy.

### Evaluation vs improvement

First, let's give some definitions to policy evaluation/improvement. These terms come from the steps of policy iteration, the foundation for many RL methods. In policy iteration, you repeat two steps until a policy stops improving.

1. Evaulate $Q^\pi(s, a)$ for your current policy $\pi$ for all state-action pairs.
2. Improve your policy $\pi$ by updating it to maximize $Q(s, \pi_\theta(s))$ for each state.

In the first evaluation step, we evaluate the Q-function for the given policy. It is worth noting that we don't have to _explicitly_ model $Q^\pi$. There are other approached we could take such as explicitly modeling the state value function $V^\pi(s)$, and then _implicitly_ derive $Q^\pi$ with observed transitions. Alternatively, we could explicit model $V^\pi$ and the environment transition dynamics $T(s' | s, a)$ from which we could derive $Q^\pi$.

Regardless of the method, "evaluation" refers to estimating a value function for a policy. If you are having difficulty understanding the exaction definition of value functions and difference between $Q$ and $V$, you may want look at my answer to [this question](../q_vs_v).

The term "improvement" regards the second step: how you make your policy better maximize the value function. As you might expect, there are many different ways to improve your policy given a value function estimate.

Because these are distinct steps, you can use different methods and data to perform them. Let's briefly review the core idea behind PPO to help explain how you might perform these steps differently.

### PPO

PPO is roughly the following algorithm.

```
Initialize parameters of state value function V.
Initialize parameters policy pi.
Do forever:
  Collect k n-length trajecories T following policy pi
  For each trajectory i to k:
    Compute Return from each step Ri1, ..., Rin
    Compute advantages Ai1, ..., Ain using Aij = Rij - V(sij)
  For M SGD steps:
    Update V(sij) toward Rij for all trajectoies and time steps
    Update policy pi using PPO CLIP objective with Aij
```

Here, the PPO CLIP objective is defined as

$$
L(s, a, \theta_\text{old}, \theta) = \min\left(\frac{\pi_\theta(a | s)}{\pi_{\theta_\text{old}}(a | s)}A(s, a), \text{clip}\left(\frac{\pi_\theta(a | s)}{\pi_{\theta_\text{old}}(a | s)}, 1-\epsilon, 1+\epsilon \right)A(s, a) \right),
$$

where $\pi_{\theta_\text{old}}$ is the policy we used to the collect the $k$ trajectories before doing any updates.

PPO is an interesting case because its evaluation method is on-policy, while its policy improvement is off-policy. That is, if you look at the above algortihm, we are updating V (over multiple steps of SGD) toward value targets of the _behavior_ policy we used to collect the data. Although we are simulatenously updating the policy, the value function is not evaluating the new updated policy, it only evaluates the behavior policy.

At the same time the behavior policy is evaluated, the policy improvement is performed multipe times through multiple steps of SGD. On the first SGD improvement step, the behavior policy and current policy match, resulting in an on-policy method update. However, after that first step in which the policy is updated, we now have a different policy than the behavior policy, resulting in the data being slightly off-policy. PPO's policy update accounts for the off-policy data in two ways. First, it uses an importance sampling ratio to correct for the difference in distributions. That's the $\frac{\pi_\theta(a | s)}{\pi_{\theta_\text{old}}(a | s)}$. Second, it clips the updates once the policy drifts too far from the behavior policy, ensuring the data it has is close enough to provide good estimates of the true policy objective. If you don't understand how the importance sampling ratio corrects for off-policy data, [see my discussion about it here](../q_learning_doesnt_need_importance_sampling/#importance-sampling).

So, although many RL methods are either off-policy or on-policy for both evaluaiton and improvement, this need not be the case. PPO is an example where the evaluation is on-policy (it evaluates the behavior policy), while the improvement step is off-policy (it improves a policy that is different than the behavior policy).

## Is on-policy or off-policy better?

There is no clear answer to whether on-policy of off-policy is better. Off-policy is a more preferable setting, because it means we can learn from any source of data, while on-policy methods are more wasteful and requires us to get more data every time we change our policy. However, at this current moment in time, on-policy methods tend to be more stable than off-policy methods. So if gathering data from your policy is cheap, you might prefer to use an on-policy method.
