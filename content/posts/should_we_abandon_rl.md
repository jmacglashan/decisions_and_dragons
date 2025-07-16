+++
title = "Should we abandon reinforcement learning?"
date = 2025-07-02T23:43:37-04:00
draft = true
+++

No, we should not abandon reinforcement learning. I get it though -- RL algorithms are brittle, difficult to scale, and often quite complicated. It's easy to understand why you might think it's the wrong approach. However, this question is predicated on a misconception. RL is not an approach. RL is a _problem definition_. It concerns agents embodied in an environment. The agent "wants" something out of its environment, but it does not know how the environment works and must learn how to get what it wants through interaction.

We don't "abandon" problems. They are imposed on us. Asking if a problem is "right" isn't a coherent question. What is a coherent question is whether a problem is _important_ or _relevant_. 

RL is an important problem. If you care about building systems that can act and learn in the world like people and animals do, the RL problem is impossible to avoid. The real question then is not whether we should abandon RL, but how we can make better algorithms to solve it.
<!--more-->

## Common misconceptions
If we hope to solve the RL problem, we must not conflate it with common methods to solve it, nor overly-specific formalisms of it. Achieving success might depend on someone coming up with a unique perspective that we haven't yet considered.

To make that more clear, let's list some things that people commonly and mistakenly think define RL.

### RL is not generalized policy iteration

> Add links to algorithms!

The most common kind of method we use to solve RL problems is _generalized policy iteration_[^1] (GPI). In GPI methods, the algorithm learns an estimate of the value function (either $Q$ or $V$) and improves its policy using it. Q-learning, SAC, PPO, etc. are all GPI methods. Even model-based RL methods like DreamerV3 and Mu-Zero have GPI sitting on top of the learned model.

GPI prevalence makes it easy to understand why you might think RL "is" GPI. But it's not. It's popular because it tends to yield the best results and it has various appealing properties. That may change in the future. I often wonder if too many of us have been seduced by GPI into a dead end. Maybe it's going to take someone very creative to shake us out our trance. Or maybe GPI will be the way to go and we just need better versions of it. We'll see.

Some examples of non-GPI methods you can use to solve RL problems include evolutionary methods, [upside down RL](https://arxiv.org/pdf/1912.02875), and the similar [decision transformer style of approach](https://arxiv.org/pdf/2106.01345). In general, you can apply almost any blackbox optimization method to an RL problem, because for as general as the RL problem is, the blackbox optimization problem is even more general.

[^1]: See Chapter 4.6 of the [Sutton Barto book](http://incompleteideas.net/book/RLbook2020.pdf).

### RL is not limited to MDPs

A Markov Decisions Process is the most common mathematical formulation used to describe RL problems. You might be excused then for thinking that RL is about solving MDPs. It's not. Not only are there non-MDP RL formulations many MDP solution methods fall more neatly into the [category of "planning" than "RL."](../model_free_vs_model_based/#planning-vs-rl)

Two other popular formalisms are partially-observable Markov decision processes (POMDPs), and stochastic games. Both have MDP-like innate structure but generalize it quite a bit. POMDPs provide a formalism for situations when the agent cannot completely observe the state and must rely on a memory of sorts and take explicit information gathering actions. Stochastic games extend the formalism to handle multiple agents acting with their own interests. Each greatly complexities, but you can also combine them together, which is when things get very interesting. So interesting, we're not even sure how to properly define the objectives. (That's a matter for another time.)

And beyond those, I'm you'll find all kinds of other variants. There is often an MDP-like flavor to the formalisms, but that's only because MDPs are pretty convenient mathematical tools. Perhaps you will come up with a better formalism though!

### RL is not model-free TD

Noting that RL is not model-free TD is redundant since we already covered that RL is not GPI, of which model-free TD is a subset. Nevertheless, I feel compelled to to be explicit because there are a some people who get stuck thinking that's what RL is. On the contrary, there is a long history of model-based RL (MBRL) work where the agent learns a model of the world, and makes decisions by planning with that model. Myself and many others are also of the mind that some form of MBRL will be how we crack the RL problem.[^2] So please, do not limit your RL exposure to model-free TD!

[^2]: There is a good chance that any future MBRL methods will still use TD, but will do so in the learned world model at least much of the time. For example, that's how DreamerV3 works. TD is useful and I'm not opposed to that so much as I am to pure model-free TD. There are also research efforts around [building different kinds of world models using TD](https://aamas.csc.liv.ac.uk/Proceedings/aamas2011/papers/A6_R70.pdf). I'm not sure that is entirely sufficient on its own, but it might be part of the story.

## Aren't you defining RL to cover everything?

Because the RL problem is quite general, some people feel like the definition is a cheap trick to count everything as RL. That is definitely not the case and it would be a mistake to see everything as an RL problem to be solved. Don't tie yourself into knots trying to formalize an environment, actions, and rewards, just so you can use an RL method. If there is a more direct method for your problem, use that method! Especially if more direct methods exploit information that you'd have to abandon to use transform the problem into something resembling an RL problem.

You should think about RL as a high-level problem description in the same way supervised learning is a high-level problem description. Like RL, supervised learning has a bevy of very different methods to solve it and various different subcommunities with different mathematical foundations. You have Bayesian methods, support vector machines, neural nets, decision trees, PAC methods, etc. Would you say everything is a supervised learning problem? Of course not. Despite that, supervised learning remains a useful high-level problem description. The same is true of RL.

## Aren't you co-opting control theory?

## How does offline RL fit in?

## The perils of language

