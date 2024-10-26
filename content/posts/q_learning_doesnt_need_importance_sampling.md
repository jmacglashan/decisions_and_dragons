+++
title = "If Q-learning is off-policy, why doesn't it require importance sampling?"
date = 2024-04-02T23:43:37-04:00
draft = false
+++

In off-policy learning, we evaluate the value function for a policy other than the one we are following in the environment. This difference creates a mismatch in state-action distributions. To account for this difference, some actor-critic methods use importance sampling. However, Q-learning, does not. There is a simple reason for that: In Q-learning, we only use samples to tell us about the effect of actions on the environment, not to estimate how good the policy action selection is. Let's make that more concrete with a simple example and re-derive the Q-learning and importance sampling approaches.<!--more-->

## Backseat gambling

<!--

Importance sampling is a method for estimating an expected value under one distribution using samples drawn from a different distribution. Classically, importance sampling was used deliberately because it turns out you can get a better estimate of an expected value using a different distribution than the one you care about[^1]. However, the main thing we're interesting in is that it let's you make estimates of expected values of one distribution with samples from another.

[^1]:
    Being able to get a better estimate using an alternative distribution turns out to be quite useful for estimating integrals, but that's another topic.

    To turn that into a concrete but problem that is more related to reinforcement learning,
-->

Let's imagine we're looking at two slot machines, which are more typically referred to as "bandits" in RL literature. Each bandit has a different payout rate that we don't know in advance. Our friend Alice told us that she is going to play from each 50-50. We watch her play for some time and observe her payouts for each pull. Later, our other friend Bob comes up and says he's going to play from from the left bandit 10% of the time and the right one 90% of the time. Let's call Alice's 50-50 strategy $\mu$ and Bob's biased strategy $\pi$. We denote their probabilities of picking the left or right arm as $\mu(a_l) = \mu(a_r) = \frac{1}{2}$ and $\pi(a_l) = \frac{1}{10}, \pi(a_r) = \frac{9}{10}$, respectively.

Can we use our observations of Alice playing to predict Bob's average payout? Yes, and there are multiple ways to do this (what a surprise, there are multiple ways to do this in RL too!).

## Separate averages

The first way that you probably are thinking of solving this question is to compute the averge payout Alice receives on the left bandit and separately compute average payout Alice receives on the right. Let's denote these averages by

$$
\begin{align*}
Q(a_l) = \frac{1}{N_l} \sum_i^{N_l} r_{li}, \\\
Q(a_r) = \frac{1}{N_r} \sum_i^{N_r} r_{ri}.
\end{align*}
$$

where $N_l, N_r$ are the number of times the left and right bandit were played, respectively and $r_{li}, r_{ri}$ are the ith payout for the left and right bandit, respectively.

After computing our averages, we can estimate[^3] Bob's expected payout by just taking a weighted average of those values, where the weights are the probability Bob woud pick each bandit:

$$
E_{\pi}\left[R \right] \approx \pi(a_l) Q(a_l) + \pi(a_r) Q(a_r)
$$

And now we have our prediction for Bob's expected payout!

[^3]: This is expression approximately equals the expected payout because our Q-values are sample averages of the bandit payouts, not the true averages.

### An iterative way to estimate expected values

In the above example, we estimated the expected value for the left and right arms by taking the arithmetic mean of the samples for each. An alternative iteravite way to estimate the expected value is to start with an arbitrary estimate and update it toward the value of each new sample. We can formalize this update rule as:

$$
Q_{t+1}(a_t) \gets Q_t(a_t) + \alpha_t (r_t - Q_t(a_t))
$$

where $a_t \in \\{l, r\\}$ is the bandit arm pulled on the $t$th pull, $r_t$ is the payout we observe at step $t$, and $\alpha_t \in (0, 1)$ is our learning rate at time step $t$. As long as $\alpha_t$ is slowly decreased (with mild constraints regarding how), the estimate will converge to the true expected value.

In RL, and ML more broadly, this style of estimating expected values is often preferred because it works well with function approximation and has some other nice properties like smoothly changing and letting you start from good guesses.

At this point it should be clear to that if you simplified Q-learning to the bandit setting where there are no states, this iterative approach to estimating separate expected values for each arm/action is what Q-learning would become. And from it, we could compute the expected value for any policy by simply taking the weighted sum of the Q-values for the policy we're evaluating.

## Importance sampling

In importance sampling, we do not have to maintain separate averages for each choice. Instead, we are just going to modify an average over _all_ our data with some weights. Spoiler, we modify it by weighting them by $\frac{\pi(a_i)}{\mu(a_i)}$ where $i$ is set to the arm that was pulled on that sample.

To show how we determine those weights let's start with how we would exactly compute Bob's expected payout if we knew the probabilities for each possible payout for each bandit. We'd ideally compute:

$$
E_{\pi} \left[ R \right] = \sum_{a \in \\{a_l, a_r\\}} \pi(a) \sum_{r\in R} p_a(r) r,
$$

where $p_a(r)$ is the probability bandit $a \in \\{a_l, a_r\\}$ will payout $r$.[^2] That is, we have a joint distribution over how Bob chooses bandits and the payout distribution of each bandit.

There's just one problem: while we know the probability values $\pi(a_l)$ and $\pi(a_r)$, we do not know the probabilities $p_a(r)$. We only get to observe samples of pulls from each bandit.

[^2]: We're going to assume there are a discrete number of possible payouts, but what we're going to do would work for continuous values if we swapped the sum for an integral

If we were directly observing Bob play, this wouldn't be so bad. We would estimate the expected value with an average of his observed payouts. We wouldn't even have to keep track of which arm he pulled, because we can think of his random choice of the bandit as part of the underlying joint distribution. But we don't get to observe Bob play. We're trying to predict his payout from observation of Alice.

Now comes the surprisingly simple importance sampling trick. Let's modify that expression without changing its value by multiplying and dividing by the probability that Alice will select the arm:

$$
\begin{align*}
\sum_{a \in \\{a_l, a_r\\}} \pi(a) \sum_{r\in R} p_a(r) r &= \sum_{a \in \\{a_l, a_r\\}} \frac{\mu(a)}{\mu(a)} \pi(a) \sum_{r\in R} p_a(r) r
\end{align*}
$$

This is obviously true. Multiplying and dividing by the same value immediately cancels itself. How could this possibly be useful? If we re-arange some terms though, a useful poperty emerges:

$$
\begin{align*}
\sum_{a \in \\{a_l, a_r\\}} \pi(a) \sum_{r\in R} p_a(r) r &= \sum_{a \in \\{a_l, a_r\\}} \frac{\mu(a)}{\mu(a)} \pi(a) \sum_{r\in R} p_a(r) r \\\
&= \sum_{a \in \\{a_l, a_r\\}} \mu(a) \frac{\pi(a)}{\mu(a)} \sum_{r\in R} p_a(r) r \\\
&= \sum_{a \in \\{a_l, a_r\\}} \mu(a) \sum_{r\in R} p_a(r) \frac{\pi(a)}{\mu(a)} r \\\
&= E_{a \sim \mu} \left [ \sum_{r\in R} p_a(r) \frac{\pi(a)}{\mu(a)} r \right] \\\
&= E_{a \sim \mu} \left [ E_{r \sim p} \left[ \frac{\pi(a)}{\mu(a)} r \right] \right] \\\
\end{align*}
$$

Ahah! We turned our expected value of Bob's distribution $\pi$ into an expected value of Alice's distribution $\mu$! We just had to weigh our payout values $r$ by $\frac{\pi(a)}{\mu(a)}$. Now we're back to a expected values of a joint distribution and we can approximate it with the mean of our samples:

$$
E_{a \sim \mu} \left [ E_{r \sim p} \left[ \frac{\pi(a)}{\mu(a)} r \right] \right]
    \approx \frac{1}{N} \sum_t^N \frac{\pi(a_t)}{\mu(a_t)} r_t
$$

where $N$ is the total number of arm pulls Alice made between both bandits, $a_t$ indicates the bandit arm she pulled on the $t$th try (either the left or right one), and $r_t$ is the payout she received.

Armed with this expression, we can approximate Bob's expected payout using our observations of Alice playing. We also could adopt the iterative estimate, rather than this mean estimate, of expected values, if we wanted.

## Where's the exepcted value under the policy in $Q^*$?

We now understand why Q-learning for simple bandits does not require importance sampling: we maintain separate expected value estimates for each action. For any given policy $\pi$ we can estimate the expected payout by taking a weighted average of the Q-values under that $\pi$. We also understand that if we didn't keep seprate estimates for each action that importance sampling would be useful to correct for the mismatch is sampling distributions.

But what about more general MDPs where there are sequential states and your value depends on what the expected value of your policy is for each subsequent state? Well, we now know how to solve that exact problem. Since Q-learning keeps separate estimates for each action from each state, we can the take a weighted average of future Q-values under the policy we care about (for Q-learning, the optimal policy) and never have to worry about importance sampling.

To illustrate that, recall that the recursive definition of the optimal Q-function is

$$
Q^\*(s, a) = R(s, a) + \gamma \sum_{s'} T(s' | s, a) \max_{a'} Q^\*(s', a').
$$

Let's pause and ask ourselves: "where in this expression is the policy we're evaluting being used?" It's all the way to the right:
$\max_{a'} Q^\*(s', a')$ is the part of the expression that is defining the policy we're evaluating. That might not look like a policy to you because we're just taking a max of values. To remedy that, let's write down the Q-function for any arbitrary policy $\pi$, not just the optimal policy:

$$
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} T(s' | s, a) \sum_{a'} \pi(a' | s') Q^\pi(s', a')
$$

Now it's obvious in that expression. Next, observe that we can express the optimal policy as a "stochastic" policy that with probability 1 selects the action with the highest Q-value[^5]:

$$
\pi^\*(a | s) = \begin{cases}
1 & \mathrm{if\ } Q^\*(s, a) = \max_{a'}Q^\*(s, a') \\\
0 & \mathrm{otherwise}
\end{cases}
$$

With that in mind, it follows that we can simplify the more general expression

$$
\sum_{a'} \pi^\*(a' | s') Q^\*(s', a') = \max_{a'} Q^\*(s', a')
$$

So really, that max operator has been an expected value of a policy distribution all along. It's just a simplification for the optimal policy case. And once you make that connection, you can see it handles the off-policy problem by using our separate averages approach.

## Q-learning: importance sampling free

Things should be getting clearer, but a recursive definition is not an algorithm and you might wonder if something is going to trip us up as move to algorithms for estimating Q. It won't, but let's be a bit more sure of that.

The recursive definition of the Q-function is neat on its own, but there is a really cool property that if you replace $Q^\*(s, a)$ with estimates and iteratively update the estimates using the recursive relationship, the values will converge to $Q^\*$! That is, using the update rule

$$
Q_{t+1}(s, a) \gets R(s, a) + \gamma \sum_{s'} T(s' | s, a) \max_{a'} Q_t(s', a')
$$

for all states and actions, we have that $Q_t \rightarrow Q^\*$ as $t \rightarrow \infty$. This algorithm is know as value iteration (VI). In this post, I won't show why VI has that property, but let's simply accept this awesome property as true for the time being.

"I asked about Q-learning, not value iteration!" I hear you yelling. I know, I know. However, Q-learning is a lot like value iteration, except Q-learning is meant to solve _reinforcement learning_ problems.

In RL, the agent doesn't know the function $R(s, a)$ nor $T(s' | s, a)$, much like how in our bandit example we didn't know the probability distributions of payouts of the bandits $p$. But the agent can interact with the environment and obsereve the outcomes.

To handle this lack of knowledge of $R$ and $T$, Q-learning uses the iterative method for estimating expected values from samples:

$$
Q_{t+1}(s, a) \gets Q_t(s_t, a_t) + \alpha (r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t))
$$

Note that the samples are drawn from the reward and transition function, _not_ the policy we're evaluating. In the one place we have to compute something about the policy: $\max_{a'} Q_t(s_{t+1}, a')$, we are using the seprate averages approach. That is, the Q-function gives us a seperate expected value estimate for each action (in each state) and when we want to evaluate the expected average for some arbitrary policy (in this case, the optimal one), we just take the weighted combination of them.

[^5]: Or if there are ties for the highest Q-value, an optimal policy is any division of the probability between the actions that tie for the hightest value.

To summarize, even though Q-learning uses samples to estimate expected values, it doesn't need importance sampling because the samples inform us about the transition dynamics only. We use the separate averages approach to compute the expected value of the optimal policy.

## Why off-policy actor critic methods use importance sampling

After all this focus on Q-learning, you may be confused about why off-policy actor critic algorithms need importane sampling! Not all actually do. It usually depends on whether the actor-critic algorithm is estimating Q-functions, which have separate estimates for each action, or state value functions, which do not.

To briefly review, one way we can define the state value function is in terms of $Q$:

$$
V^\pi(s) = \sum_a \pi(a | s) Q(s, a)
$$

If the algorithm estimates the Q-function, it can recover V from Q allowing it to be off-policy much like Q-learning is. And many off-policy actor-critic algorithms do just that. Examples include SAC, TD3, and DDPG.

However, other off-policy actor-critic algorithms, like IMPALA[^6] do not. Instead, they directly use sampled returns to estimate $V$. Consequently, they need to correct for the sample distribution difference with importance sampling, much like we did in our bandit example.

[^6]: IMPALA doesn't actually use the standard importance sampling correction ratios we discussed. They use an alternative sample weighting that tends to mitigate problems with compounding probabilities.

## Why wouldn't we always learn Q-values?

You might wonder why we don't always estimate Q-values if we want to do off-policy learning. Afterall, it was probably the simpler approach you first imagined when I described the simple bandit problem. It also has the nice property that you don't have to know what the probabilities of the behavior policy were (e.g., Alice's policy $\mu$ in our bandit example). You only have to know the probabilities of the policy you want to evaluate.

However, there are still some nice things about using state value estimates. First, if your action space is very large, maintaining separate estimates for each action can become problematic. If you're using function approximation, you might try to avoid that problem by generalizing over the actions. That is in fact what SAC, TD3, and DDPG do. But if you're introducing funtion approximation across your actions, now you've opened the door for more biased estimates for each action. Furthermore, you can only really do one-step updates where you bootstrap from the next state's Q-values and that adds another source of bias. These sources of bias are not trivial -- very often if algorithms like SAC fall apart it's inherently linked to bias issues in the estimate of the Q-function. For these reasons, estimating the state value function and using importance sampling may be preferable.
