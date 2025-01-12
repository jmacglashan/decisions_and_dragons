+++
title = 'What is the "horizon" in reinforcement learning?'
date = 2024-04-21T16:49:00-04:00
draft = false
+++

In reinforcement learning, an agent receives reward on each time step. The goal, loosely speaking, is to maximize the future reward received. But that doesn’t fully define the goal, because each decision can affect what reward the agent can receive the future. Consequently, we’re left with the question "how does potential future reward affect our decision right now?" The "horizon" refers to how far into the future the agent will optimize its reward. You can have finite-horizon objectives, or even infinite-horizon objectives.

<!--more-->

## I'll be taking my marshmallow now

One answer to our question is the future doesn't matter! Our objective is to only maximize the immediate reward received for the next action. [You want your marshmallow, and you want it now](https://en.wikipedia.org/wiki/Stanford_marshmallow_experiment).

Such an objective is a "finite horizon" objective, where "horizon" refers to how many decision steps into the future the agent cares about the reward it can receive. In this case, we’ve defined a 1-step horizon objective, the most myopic objective you can define, since it only cares about how big the next immediate reward is.

## I can wait, for a time

The one-step horizon has an obvious limitation. Maybe if you sacrifice a little bit of reward now you can have a bigger total reward later. If we want the agent to care about the future, we need a longer horizon.

Fortunately, the one-step horizon is easy to generalize to a longer horizon. We could define a 2-step horizon, in which the agent makes a decision that will maximize the total reward it will receive in the next 2 time steps. Or we could choose a 3, or 4, or n-step horizon!

In this finite-horizon regime, the agent cares about how well it can maximize total reward for all steps within the horizon, but after that point, it stops trying to optimize for more reward.

Finite horizons are perhaps the most common objective you'll see in control theory literature. For example, in model predictive control, the agent optimizes what it can do for the next $n$ time steps. Then, after taking an action, the agent again optimizes what it could do for the next $n$ time steps, looking one step further than on the last step since its moved forward one step in time. This approach is sometimes called "receding horizon" because on each step, the horizon recedes back one step further, always keeping the horizon out of reach.

## I have infinite patience, but I'd prefer it now

A finite horizon, even a long finite horizon, does have its limitations of course. You have to pick a somewhat arbitrary threshold, and that choice has sharp consequences. If you choose a horizon of 15, but this causes the agent to miss the opportunity to receive a massive reward on the 16th step, too bad. You chose 15, and that's what it will optimize for.

We might ask ourselves: is there a way to tell the agent to optimize for an _infinite_ horizon?

Optimizing for an infinite horizon can get a little funky. If we summed up all rewards infinitely far into the future, a policy that generates positive reward indefinitely would have have infinite total reward. Infinite values complicate matters because it makes it impossible to compare many policies. That is, if policy $a$ generated more reward on every time step than policy $b$, but both generated infinite total reward, we would be unable to say that policy $a$ was better than $b$.

One solution to this enigma is to use a _discounted_ infinite horizon objective: the most common objective in RL literature. In an infinite-horizon discounted objective, we sum up each reward, but discount how much we care about each reward by how far into the future it is. Using a discount parameter $\gamma \in [0, 1)$, we define the objective to be:

$$
r_1 + \gamma r_2 + \gamma^2 r_3 + \gamma^3 r_4 + ...
$$

Because each possible reward is geometrically decreased, we ensure the total value of any infinite future is always finite (assuming all rewards are bounded). When comparing two possible futures that have the same undiscounted reward total, the one that achieves reward faster will win out. That is, this discounted objective prefers getting lots of reward sooner rather than later.

Discounting solves the dilemma of missing the opportunity to achieve a big reward just one or a few steps later than a finite horizon limit, because it accounts for all future reward. However, it does impose a kind of "soft horizon." The closer The discount is to zero, the more myopic it will be, with it behaving exactly like a one-step finite horizon when $\gamma = 0$.

## All moments are equally good, but I've got a math problem for you

The discounted objective is an infinite horizon objective, but the discount factor $\gamma$ acts a lot like a horizon. You might wonder if we can do better while avoiding the problem of infinite-value returns.

And we can! Sort of. That is, you can instead consider optimizing the _average reward_ over the entire infinite future. There are even constraints you can add to that objective such that if two futures have the same average reward, you prefer the one that gathers reward the fastest. See the [R-learning paper](https://www.researchgate.net/profile/Anton-Schwartz/publication/221346025_A_Reinforcement_Learning_Method_for_Maximizing_Undiscounted_Rewards/links/5e72421aa6fdcc37caf4cf4b/A-Reinforcement-Learning-Method-for-Maximizing-Undiscounted-Rewards.pdf) for a discussion of that.

Unfortunately, designing algorithms that optimize the average-reward objective has proven challenging. There is still work in the area, so we may yet find a practical solution. In fact, I'm increasingly optimistic that a solution is within reach. Perhaps when we do, we'll all switch to using average reward. Until then though, most work sticks with the discounted objective.
