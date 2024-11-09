+++
title = 'What is the difference between model-based and model-free RL?'
date = 2024-03-29T19:50:17-04:00
draft = false
+++

In reinforcement learning, the agent is not assumed to know how the environment will be affected by its actions. Model-based and model-free reinforcement learning tackle this problem in different ways. In model-based reinforcement learning, the agent learns a model of how the environment is affected by its actions and uses this model to determine how to act. In model-free reinforcement learning, the agent learns how to act without ever learning to precisely predict how the environment will be affected by its actions. <!--more-->

To better understand this distinction, it is helpful to revisit the definition of an MDP, how to solve it, and how RL makes this problem harder.

## MDPs

Let's revisit the components of an MDP, the most typical decision-making framework for RL. An MDP is defined by a 4-tuple $(S, A, R, T)$ where,

- $S$ is the state/observation space of an environment,
- $A$ is the set of actions the agent can choose between,
- R(s, a) is a function that returns the reward received for taking action $a$ from state $s$,
- $T(s' | s, a)$ is a transition probability function, specifying the probability that the environment will transition to state $s'$ if the agent takes action $a$ in state $s$.

Our goal is to find a policy $\pi$, mapping states to action choices we should make in each state, that maximizes the expected future (discounted) reward.

## Planning vs RL

If we know what all those elements of an MDP are, we can find a good policy before ever actually executing an action in the environment. In AI, we typically call computing the solution to a decision-making problem before executing an actual decision "planning." Some classic planning algorithms for MDPs include Value Iteration, Policy Iteration, Monte Carlo Tree Search, and whole lot more.

But the RL problem isn’t so kind to us. What makes a problem an RL problem, rather than a planning problem, is the agent does _not_ know all the elements of the MDP, preventing it from planning.[^1] Specifically, the agent does not know how the environment will change in response to its actions (the transition function $T$), nor what immediate reward it will receive for doing so (the reward function $R$). The agent will simply have to try actions in the environment, observe what happens, and somehow, find a good policy from doing so.

[^1]: Beware: I'm entering into controversial territory here! There are methods that use knowledge of the transitions dynamics that some people will still call "RL." Like most things, lines get blurry. For example, the authors of [AlphaGo](https://deepmind.google/research/breakthroughs/alphago/) called their method "RL" even though they used MCTS and the dynamics of the game were known and provided to the algorithm. Usually, when people call "planning" methods "RL" methods, they do so because the method makes limited use of its knowledge of the transition dynamics. Typically for computational reasons. Nevertheless, methods like MCTS require the agent to have knowledge of how the environment is affected by its actions, and they exploit this knowledge. For this reason, I still call them "planning" methods. Call me a curmudgeon if you must.

How can an agent find a good policy if it does not know the transition function $T$ nor the reward function $R$? It turns out there are lots of ways!

## Model-based RL

One approach that might immediately strike you is for the agent to learn a model of how the environment works and then plan a solution using that model. That is, suppose the agent is currently in state $s_1$, takes action $a_1$ and then observes that the environment transitions to state $s_2$ with reward $r_2$. That observation can be used as training data to predict $T(s_2 | s_1, a_1)$ and $R(s_1, a_1) \rightarrow r_2$ with supervised learning.[^3] Using its learned models of $T$ and $R$, the agent can plan a solution. And as the the models get better, it can form more accurate plans. RL solutions that follow this framework are model-based RL algorithms.

[^3]: Training the transition function predict $T(s_2 | s_1, a_1)$ can be a little complicated, because $T(s_2 | s_1, a_1)$ is meant to represent the probability of reaching state $s_2$. If we can assume the environment is deterministic with small levels of noise, we can train a function to predict $s_2$ given $s_1$ and $a_1$, and assign its output probability 1. If the environment is more stochastic, things get a little harder. However, generative ML is increasingly getting better. While generative models often don't provide exact probabilities, there are many planning methods that only require a generative model and do not need a full probability distribution.

## Model-free RL

As it turns out though, we don’t have to learn a model of the environment to find a good policy. One of the most classic examples is Q-learning, which directly estimates the optimal Q-values of each action in each state.[^2] A policy may then be derived from the learned Q-values by choosing the action with the highest Q-value in the current state. Other model-free methods include actor-critic and policy search methods, which search over the policy space to find policies that result in better reward from the environment.

Because these approaches do not learn a model of the environment they are called model-free algorithms. Model-free methods demonstrate that you do not need to know what the next state for any given action is to act well. You just need to be able to estimate which actions in which states will allow you to collect a lot of future reward.

[^2]: A Q-value $Q(s, a)$ is an estimate of how much long-term reward the agent expects to get if it takes action $a$ from state $s$. If you don't feel comfortable with the definition of a Q-value, see [my answer to this question](../q_vs_v/).

## Is one approach better than another?

Answering which is better is rather complicated. Model-free methods may be less biased because they rely more closely on real experiences. It turns out it's hard to learn good models of the environment, and if your model used for planning is poor, you may make bad decisions. However, model-free methods tend to require a lot of interaction with the environment and have some tricky algorithm stability issues. For the time being, you will just have to figure out what works best for your environment.

## A simple heuristic

If you want a way to check if an RL algorithm is model-based or model-free, ask yourself this question: after learning, can the agent make predictions about what the next state and reward will be before it takes each action? If it can, then it’s a model-based RL algorithm. if it cannot, it’s a model-free algorithm.

This same idea may also apply to decision-making processes other than MDPs.
