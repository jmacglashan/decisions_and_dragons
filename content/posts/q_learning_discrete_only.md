+++
title = "Why doesn't Q-learning work with continuous actions?"
date = 2024-04-21T15:36:15-04:00
draft = false
+++

Q-learning requires finding the action with the maximum Q-value in two places: (1) In the learning update itself; and (2) when extracting the policy from the learned Q-values. When there are a small number of discrete actions, you can simply enumerate the Q-values for each and pick the action with the highest value. However, this approach does not work with continuous actions, because there are an infinite number of actions to evaluate!<!--more-->

## Examining the Q-learning update

Lets take a look at the core Q-learning update rule:

$$
Q(s, a) \gets Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right],
$$

where $r$ is the observed reward and $s'$ is the next observed state.

Or if we’re working with function approximation of the Q function (e.g., using a neural net), we can frame this as a loss minimization problem where the loss function to minimize is:

$$
L(\theta) \triangleq \frac{1}{2}\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2,
$$

where $\theta^-$ is a copy of the parameters $\theta$ that we do not differentiate through (and in the case of DQN and others, may be many updates old).

Effectively we’re doing a standard supervised regression problem on our function approximation of Q, except instead of having pre-defined labels in a dataset, we compute a “label” on each step with the value:

$$
y \triangleq r + \gamma \max_{a'} Q_{\theta^-}(s', a').
$$

You might think that as long as our Q-function network operates on continuous actions we should be alright. After all, neural networks train on all kind of continuous inputs, why not the action?

Except how are you going to compute that label? Do you see the problem? It’s that damn $\max$ operator!

If your actions are continuous, and therefore infinite, you can’t just look up which action produces the maximum q-value because there are an infinite number of actions to scan.[^1]

## Extracting the policy is no easier

This same problem emerges when we want to extract the policy from the Q-values, because the greedy policy from the learned Q-values (which is optimal when the optimal Q-values are learned) is:

$$
\pi(s) = \arg\max_a Q(s, a).
$$

So when we want to act in the world, here too does the curse of the max operator thwart us. Even if you use epsilon-greedy, the greedy part of the policy is going to require you to find the action with the maximum Q-value!

## Alternatives

Taking a step back, we _mostly_ already know how to solve this problem in AI: finding the the max is just an optimization problem! Therefore, you can just replace the max operator with your favorite optimization algorithm. For example [QT-OPT](https://arxiv.org/abs/1806.10293), uses the cross entropy method to approximate the max. Or you could use SGD on the Q-function to find the maximum action.

However, if you’re going to replace the exact max with the result of some optimization algorithm you’re

1. probably increasing the compute time for both training (you’re doing optimization in the inner loop of an optimization algorithm!) and inference/acting; and
2. only going to end up with an approximate max anyway which, depending on details, may get stuck in local optima.[^2]

[^2]: Unless your Q-function is convex over the actions or has some other really convenient, but restrictive, properties.

That’s not to say those approaches can’t be successful, but it’s a fairly significant deviation from the classic Q-learning algorithm that will affect the solution conept the algorithm finds. I'll let you decided whether to still call algorithms that use these modifications Q-learning.

Personally, I prefer actor-critic methods for continuous actions since they do not suffer this same problem. And if you want an actor-critic method that is quite similar to Q-learning, you may want to consider [DDPG](https://arxiv.org/abs/1509.02971), [TD3](https://arxiv.org/abs/1802.09477v3), or [SAC](https://arxiv.org/abs/1801.01290). If you want to learn more about how DDPG (and largely TD3) works, you may want to see [my post](../ddpg_grad/) on its policy gradient derivation.

[^1]: Okay, _nothing_ is actually continuous in computers. Computer numbers are actually discrete and we just use clever discrete encodings like 32-bit [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) to represent a large (but finite!) set of fractional values. But there's still a hell of lot of possible values that you wouldn't want to enumerate! Like, over 4 billion. And that's if you only have _one_ continuous action dimension.
