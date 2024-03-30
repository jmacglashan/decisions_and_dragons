+++
title = 'What is the difference between model-based and model-free RL?'
date = 2024-03-29T19:50:17-04:00
draft = false
+++

In reinforcement learning, the agent is not assumed to know how the environment will be affected by its actions. Model-based and model-free reinforcement learning tackle this problem in different ways. In model-based reinforcement learning, the agent learns a model of how the environment is affected by its actions and uses this model to determine good behavior. In model-free reinforcement learning, the agent learns how to act well without ever learning to predict precisely how the environment will be affectd by its actions. 

To understand these two paradigms more completely, it is helpful to revisit the definition of an MDP, how to solve it, and how RL specifically makes this problem harder.

## MDPs
More specificially, lets revisit the components of an MDP, the most typical decision making framework for RL. An MDP is defined by a 4-tuple $(S, A, R, T)$ where,
* $S$ is the state/observation space of an environment,
* $A$ is the set of actions the agent can choose between,
* R(s, a) is a function that returns the reward received for taking action $a$ from state $s$,
* $T(s' | s, a)$ is a transition probability function, specifying the probability that the environment will transition to state $s'$ if the agent takes action $a$ in state $s$.

Our goal is to find a policy $\pi$ that maximizes the expected future (discounted) reward.

## Planning vs RL
If we know what all those elements of an MDP are, we can compute the solution before ever actually executing an action in the environment. In AI, we typically call computing the solution to a decision-making problem before executing an actual decision "planning." Some classic planning algorithms for MDPs include Value Iteration, Policy Iteration, Monte Carlo Tree Search, and whole lot more.

But the RL problem isn’t so kind to us. What makes a problem an RL problem, rather than a planning problem, is the agent does *not* know all the elements of the MDP, precluding it from being able to plan a solution. Specifically, the agent does not know how the environment will change in response to its actions (the transition function  $T$), nor what immediate reward it will receive for doing so (the reward function $R$). The agent will simply have to try taking actions in the environment, observe what happens, and somehow, find a good policy from doing so.

So, if the agent does not know the transition function $T$ nor the reward function $R$, preventing it from planning a solution out, how can it find a good policy? Well, it turns out there are lots of ways!

## Model-based RL
One approach that might immediately strike you is for the agent to learn a model of how the environment works from its observations and then plan a solution using that model. That is, if the agent is currently in state $s_1$, takes action $a_1$, and then observes the environment transition to state $s_2$, with reward $r_2$, that information can be used to improve its estimate of $T(s_2 | s_1, a_1)$ and $R(s_1, a_1)$, which can be performed using supervised learning approaches. Once the agent has adequately modelled the environment, it can use a planning algorithm with its learned model to find a policy. RL solutions that follow this framework are model-based RL algorithms.

## Model-free RL
As it turns out though, we don’t have to learn a model of the environment to find a good policy. One of the most classic examples is Q-learning, which directly estimates the optimal Q-values of each action in each state (roughly, the utility of each action in each state), from which a policy may be derived by choosing the action with the highest Q-value in the current state. Other model-free methods in actor-critic and policy search methods, which directly search over the policy space to find policies that result in better reward from the environment. Because these approaches do not learn a model of the environment they are called model-free algorithms. Model-free methods take the position that you do not really need to know what the next state is to act well, you just need to be able to keep track of which actions in which states will allow you to collect a lot of future reward and take the actions that will generate the most.

## Is one approach better than another?
Answering which is better is rather complicated. Both have their advantages. Model-free methods may end up less biased because they rely more closely on their direct experiences. It turns out to be hard to learn good models of the environment, and if your model used for planning is poor, you may make bad decisions. However, model-free methods tend to need quite a lot of interaction with the environment and have some tricky algorithm stability issues, which may make model-based RL methods more appealing. For the time beign, you will just have to figure out what works best for your environment.

## A simple heuristic 
If you want a way to check if an RL algorithm is model-based or model-free, ask yourself this question: after learning, can the agent make predictions about what the next state and reward will be before it takes each action? If it can, then it’s a model-based RL algorithm. if it cannot, it’s a model-free algorithm.

This same idea may also apply to decision-making processes other than MDPs.