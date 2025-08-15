+++
title = "Should we abandon RL? Is it the right approach?"
date = 2025-07-02T23:43:37-04:00
draft = false
+++

No, we should not abandon reinforcement learning. I get it though — RL algorithms are brittle, difficult to scale, and complicated. However, this question is predicated on a misconception. RL is not an approach. RL is a _problem definition_.

RL is the problem of determining how an agent should make decisions in an unfamiliar environment. It must act to both learn about its environment and pursue its objective. It must learn from its experiences, rather than a human-curated dataset.

We don't abandon problems. They are imposed upon us and we ignore them at our peril. Asking if a problem is "right" isn't a coherent question. What is a coherent question is whether a problem is _important_ or _relevant_. 

RL is an important problem. If you care about building systems that can act and learn in the world like people do, then the RL problem is impossible to avoid. The question is not whether we should abandon RL, but how we can make better algorithms to solve it.
<!--more-->

## Who are you to say what RL is?
I've been deep in this field long enough to have a meaningful opinion on the matter. But my opinion that RL is a problem, rather than an approach, isn't controversial among RL researchers. If you don't want to take it from me, then how about from Rich Sutton and Andy Barto?

[In the first edition of their RL book, Chapter 1.1](http://incompleteideas.net/book/first/ebook/node7.html), they say:
> Reinforcement learning is defined not by characterizing learning methods, but by characterizing a learning _problem_. Any method that is well suited to solving that problem, we consider to be a reinforcement learning method.

You can read on to see that what I've described in this post aligns with their comments.

## Common misconceptions
If we hope to solve the RL problem, we must not conflate it with common methods to solve it, nor with overly-specific formalisms of it. Achieving success might depend on someone coming up with a unique perspective that we haven't yet considered.

To avoid that pitfall, let's list some things that people mistakenly think define and limits RL.

### RL is not limited to model-free methods
RL is not limited to model-free methods. That is, it's not limited to [Q-learning](https://link.springer.com/article/10.1007/BF00992698), [SARSA](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems), their derivatives, nor actor-critic and policy-gradient methods. While the RL problem requires the agent to act without initially knowing how its actions will affect the environment, there is nothing about the RL problem that prevents the agent from [learning a world model and planning with it](../model_free_vs_model_based/). In other words, model-based RL (MBRL) methods still solve the RL problem.

There are, however, good reasons why model-free RL has thus far been predominant. Learning a _useful_ world model turns out to be challenging. And worse, choosing the right kind of world-model training objective can be tricky. Asking the algorithm to maximize feature-level accuracy doesn't tend to make useful trade offs. While sufficiently accurate feature predictions will allow an agent to plan well, in practice even small errors tend to result in a poor model for planning. What we really want is an objective to make good predictions of future rewards conditioned on different counterfactual actions. We want something like a Q-function loss that is estimated through a world model. A practical implementation of that is harder than it sounds though.

Fortunately, algorithms like [DreamerV3](https://arxiv.org/abs/2301.04104) and [Mu-zero](https://arxiv.org/abs/1911.08265) show that MBRL is getting better. And with the broader AI community taking an interest in world models (albeit in a dataset-driven, rather than experience-driven, paradigm), there are more people working on this problem than ever before. 

Myself and many others believe that doing MBRL right may be key to cracking the RL problem. So please, do not limit your RL research to model-free methods!

### RL is not limited to GPI

The most common kind of method we use to solve RL problems is _generalized policy iteration_[^1] (GPI). In GPI, the algorithm estimates the value function (either $Q$ or $V$) and uses it to improve its policy. [Q-learning](https://link.springer.com/article/10.1007/BF00992698), [SAC](https://arxiv.org/abs/1801.01290), [PPO](https://arxiv.org/abs/1707.06347), etc. are all GPI methods. Even model-based RL methods like [DreamerV3](https://arxiv.org/abs/2301.04104) and [Mu-Zero](https://arxiv.org/abs/1911.08265) have GPI sitting on top of the learned world model.

GPI is popular because it tends to yield the best results and it has various appealing properties. However, just as the RL problem does not restrict us to model-free methods, it does not restrict us to GPI methods either.  

Furthermore, the prevalence of GPI may only be temporary. I sometimes wake up in a cold sweat wondering if too many of us have been seduced by GPI into a dead end. Maybe it will take a creative person to free us. Or maybe GPI is the way to go and we just need better versions of it. We'll see.

Some examples of non-GPI methods include evolutionary methods, [upside down RL](https://arxiv.org/pdf/1912.02875) and the similar [decision transformer style of approach](https://arxiv.org/pdf/2106.01345), and meta RL methods, like [AdA](https://arxiv.org/pdf/2301.07608), in which the agent learns to learn. You can also apply almost any blackbox optimization method to an RL problem because the blackbox optimization problem is even more general than the RL problem.

[^1]: See Chapter 4.6 of the [Sutton Barto book](http://incompleteideas.net/book/RLbook2020.pdf).

### RL is not limited to MDPs

The environment in an RL problem is typically formalized as a Markov Decision Process (MDP). You may be excused then for thinking that RL is about solving MDPs, but it's not. Not only are there non-MDP RL formulations, many MDP-based problems fall more precisely into the [category of "planning" than "RL."](../model_free_vs_model_based/#planning-vs-rl)

Two other popular formalisms are partially-observable Markov decision processes (POMDPs) and stochastic games. Both generalize MDPs. POMDPs extend MDPs to situations when the agent cannot completely observe the state. The agent must rely on a memory and take explicit information gathering actions, even after it's familiar with the environment. Stochastic games extend MDPs to include multiple agents acting on their own interests that may or may not align with each other.

Both POMDPs and stochastic games are harder to solve than MDPs, but you can combine them if you want something even harder (and frankly, more accurate to reality). 

You'll find other variants beyond those too. Many have an MDP flavor because MDPs are pretty convenient mathematical tools. But perhaps you will come up with a very different and better formalism.

## Isn't this definition vague and all encompassing?

Because the RL problem is general, some people feel like the definition is a cheap trick to count anything as RL. That is definitely not the case. In fact, I would say most AI research is decidedly _not_ RL research. 

It would also be a mistake to see everything as an RL problem. Don't tie yourself into knots trying to formalize an environment, actions, and rewards, just so you can use an RL method. If there is a more direct method for your problem, use that method! Especially if more a direct method exploits information that you'd have to abandon to transform the problem into something resembling an RL problem.

I will concede that the definition of RL is somewhat vague. That is by design to not over commit to any particular formalism (see the previous section). However, it's no more vague than any number of other high-level problem descriptions like "supervised learning." 

Like RL, supervised learning is a vague definition by design. It allows for a bevy of different formalizations and an even greater variety of methods to solve it. You have Bayesianism, maximum likelihood, probably approximately correct, optimization, etc. You even have formalisms that use alternatives to probability theory like [Dempster–Shafer theory](https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory)!

Despite that, supervised learning remains a useful high-level problem description. The same is true of RL.

## Aren't you reinventing control theory?
Another common objection to the RL problem is that RL researchers are reinventing control theory and are otherwise unaware of it.

On the contrary, when modern RL was first being developed by Andy Barto, Rich Sutton, and others, they were well aware of control theory. According to Andy Barto in his [RLC 2024 keynote talk](https://www.youtube.com/watch?v=-gQNM7rAWP0), it was very important to him that they did not "become a cult" and that they build on existing control theory and other mathematics. The RL community was, and remains, well aware of control theory. Hell, if I had a nickel for every time I said "Bellman equation," I'd be happily retired.[^4]

[^4]: That's a lie. I'd still be working on RL. I'd just be wealthy and working on RL. I'm also not sure if I'd be happy. That's a different issue though.

You might ask, "why splinter from control theory then?" There may be several reasons, but I can't say for sure. See, I hadn't actually been alive all that long when it started. But while I can't be certain of why that happened, there are good reasons for why RL should have become a separate community from control theory.

While control theory work provides a mathematical foundation for RL, the field for the most part does not focus on the RL problem. Control theorists are much more concerned with engineering a model of the system dynamics and deriving control laws. In contrast, RL focuses on an agent that must learn about the system and how to control it on its own through interaction. 

Yes, control theory has plenty of work on [system identification](https://en.wikipedia.org/wiki/System_identification), but the way that is done often sidesteps the problems RL cares about. For example, system identification often fits the parameters of a human-engineered physics model. Even when its model is a black box, like a neural net, the data used to train it is usually assumed to be provided upfront. There isn't a focus on simultaneously acting, learning, and collecting data.

I'm not going to tell you that no one from the control theory community ever worked on the RL problem. I'm sure you can find instances that are a good fit. And today, there may be more overlap since communities cross fertilize with each other. 

The point is there wasn't a _focus_ on the RL problem in control theory communities. If you want people to focus on a problem, you're going to need to first name it and define it.

To give an analogy, Yann LeCun popularized the term "self-superivsed learning." You can make a case that the term is just a rebranding of "unsupervised learning." However, the point of introducing the term was to highlight a particular kind of learning problem that Yann thought was important enough to be called out and focused on.

Similarly, the goal of the term RL is to call out an important problem. Once called out, academic communities will form to work on it.

## How does offline RL fit in?
Offline RL does not match the definition I've given for RL. It sidesteps what I (and, I believe, the founders of modern RL) consider one of the more important dimensions of RL: experiential learning. Instead, the agent is provided with a large curated dataset upfront. That's precisely why the qualifier "offline" is used: to indicate that it's not the standard RL problem.

Even so, the scope of offline RL keeps the RL problem in view. For example, many offline RL methods train models that can be directly used in RL algorithms that learn from experience. In this way, offline RL acts similarly to a pretraining step for RL. Additionally, many use cases of offline RL are for a delayed off-policy RL paradigm in which the agent collects a lot of data, does a bunch of "offline RL," and then repeats the process.

For these reasons, I think we can accept the association. But to be clear, yes, the "offline" qualifier is indicating that it's not quite the same problem.

## The perils of language

Finally, keep in mind that natural language is not a formal language. It can be imprecise and overloaded. That is a feature of natural language, not a bug. Because of that, you will hear people use the term "RL" haphazardly. 

For example, you might hear even well-established researchers say "I used RL to train a policy," in which they use the term "RL" like it's an approach. Don't come running back to tell me I must be wrong because some big shot (or even I!) used the term that way. 

When someone says something like that, what they really mean is "the policy model is a product of solving an RL problem," but that's really wordy! The intention of the original expression ought to be clear. 

It's similarly okay to say things like "RL method" or "RL algorithm" to refer to a method that solves RL problems. If you were paying attention, you may have noticed that the term "MBRL" has "RL" in it, and is referring to an approach. That's because it's an approach that solves RL problems.

In short, it's okay to be loose with your language. What gets you into trouble is forgetting that RL is, first and foremost, a problem definition. And if that problem is important, you cannot abandon it.