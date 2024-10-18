+++
title = 'Why does the policy gradient include a log probobability term?'
date = 2024-03-29T20:07:54-04:00
draft = false
+++

Actually, it doesn't! What you're probably thinking of is the [REINFORCE](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) *esitmate* of the policy gradient. How we derive the REINFORCE estimate you're familiar with and *why* we use it is something I found to be glossed over in literature, or presented in an overly technical way, but fortunately, it is not a hard concept to learn!<!--more-->

The short answer for where the log probability comes from is that REINFORCE uses a trick to make the policy gradient look like an expected value over the gradient, and the log probability manfiests as an importance sampling correction to make this work. We use this trick because we'd like to use the samples the agent takes from each state to estimate the gradient, but computing the regular gradient without REINFORCE would require us to observe the outcome of each aciton in each state, including the ones we didn't take.

That might not make much sense yet. Let's remedy that.

## The policy objective
First, lets identify the following: what is the objective function we want to maximize? From “[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)” we have the following objective:
$$
J(\pi) = \sum_s d^\pi(s) \sum_a \pi(a | s) Q^\pi(s, a)
$$
where
* $d^\pi(s)$ defines the (discounted[^1]) probability of being in state $s$ when following policy $\pi$,
* $\pi(a | s)$ defines the probability of selecting action $a$ from state $s$ when following policy $\pi$,
* $Q^\pi(s, a)$ is our usual Q-function defining the expected future return when taking action $a$ from state $s$ and then following policy $\pi$ for all subsequent time steps.

[^1]: When solving discounted future reward problems -- the most common case in RL -- the state distribution is a little funny. In this case, it's *not* just the probability (density) of being in a state. It's the *discounted* probability. Without going into too much detail, this discounts the probability by how long it takes to reach a state and averages over all the different lengths of time it could take. One way to think about discounted probability is by noting an equivalence between solving a discounted MDP and solving a modified undiscounted MDP where there is a probability of $1 - \gamma$ of transitioning to an absorbing state at any time. In practice, algorithms often ignore this fact, but to be mathematically precise, the policy gradient for discounted MDPs use discounted state probability distributions. A more complete discussion of this topic is left for another time.

This math might feel a bit scary, but in words, the objective is the average Q-value (across all states action) when following your policy. Adjusting your policy to maximize this objective means the average Q-value goes up, which in turn means the averaged expected future return is higher, which is of course the objective of RL!

The quesiton is *how* do we optimize this objective? If we are defining our policy with a neural net or some other differentiable function approximation, we can compute gradients of the objective with respect to the policy neural net parameters $\theta$ and use something like stochastic gradient descent (SGD) to improve the policy.[^2]

[^2]: To use gradient *descent* we would want to turn this objective into a loss by multiplying it by negative one. Otherwise we can use gradient *ascent*.)

## The true policy gradient
Of course that means we need to know how to compute the gradient of that objective. The main result of the policy gradient theorem is that the derivative of the objective is the following:
$$
\frac{d}{d\theta} J(\pi) = \sum_s d^\pi(s) \sum_a \frac{d \pi(a | s)}{d\theta} Q^\pi(s, a)
$$

The neat and important property of this result is we really only need to compute the gradients of the probability that our policy will take any given action and then we can just multiply that by the Q-value (and take a weighted average over discounted state distribution). We don't need to compute gradients through transitions functions, which usually are not known in RL problems.

We do of course need an estimate of our Q-values too, and we need to take the average over our states. The tempting approach is to just use samples for these remaining bits. I.e., maybe we could just execute our polciy in our environment a bunch of times, estimate Q-values with observed returns, multiply them by the gradient of our policy, and average over the visted states?

Unfortunately, no, on its own that will not work.

## No takesies backsies
To see the problem lets ignore the state distribution part of the objective on focus on the last part:
$$
\sum_a \frac{d \pi(a | s)}{d\theta} Q^\pi(s, a)
$$
This may look like an expected value, but it's not. It's the *derivative* of an expected value. Because of that, we cannot estimate the gradient by using sampled actions. To compute the gradient correctly for each state, we'd have to know what the Q-value is for every action, and sum their gradients together.

Summing over all the actions for each state may not sound bad, but in RL, once you take an action, you have to live with the consequences. You cannot undo it and then try a different action to see what would happen. And even if we could, that's a lot of extra environment interactions! For every state we visit, we'd have to roll out another trajectory for every other possible action we could have taken. (Good luck if your actions are continuous!)

## REINFORCE to the rescue
Fortunately, REINFORCE provides us a way to turn our derivative of an expected value into an expected value of the derivative of an expected value, allowing us to estimate the gradient with samples. To achieve this, REINFORE uses an approach very similar to importance sampling. If you are not familiar with importance sampling, that's okay, because we're going to walk through the idea here.

We begin by doing something ridiculous: inside the sum, we're going to multiply everything by the probability of our policy taking the action and immediately divide by it to cancel it:
$$
\sum_a \frac{d \pi(a | s)}{d\theta} Q^\pi(s, a) = \sum_a \frac{\pi(a | s)}{\pi(a | s)} \frac{d \pi(a | s)}{d\theta} Q^\pi(s, a)
$$
So far, we've done nothing. All we did was add needless work by multiplying by the policy probability and dividing by it. But when you look at it this way, you may notice that this is the same as an expected value!
$$
\sum_a \frac{\pi(a | s)}{\pi(a | s)} \frac{d \pi(a | s)}{d\theta} Q^\pi(s, a) = E_{a \sim \pi(\cdot | s)} \left[ \frac{1}{\pi(a | s)}  \frac{d \pi(a | s)}{d\theta} Q^\pi(s, a) \right]
$$
Now we're in business: if we sample trajectories from our environment by following our policy, that will give us samples from the state and aciton distribution in our new gradient. That is, our naive hope was _almost_ right. All we had to do to correct for the fact that we weren't summing the policy gradients over each action uniformly is divide by the probability of our policy selecting the action it took.

Except I promised log probabilities and as of now they haven't shown up.

## Enter log probabilities
Fortunately, introducing the log probability is just one more step away. Let's remark on this simple calculus identity about the derivative of logs:
$$
\frac{d}{dx} \log f(x)= \frac{1}{f(x)} \frac{df(x)}{dx}
$$
Here we see that we have the right-hand-side of that in our REINFORCE estimate of the gradient. Therefore, we can simplify it
but substituing the log probability.
$$
E_{a \sim \pi(\cdot | s)} \left[ \frac{1}{\pi(a | s)}  \frac{d \pi(a | s)}{d\theta} Q^\pi(s, a) \right] = E_{a \sim \pi(\cdot | s)} \left[ \frac{d\log \pi(a | s)}{d\theta} Q^\pi(s, a) \right]
$$
In addition to simplifying the expression, using the log probabiltiy is usually preferable because it tends to be more numerically
stable for floating-point math on computers. As such, we almost always use the log probability formulation.
