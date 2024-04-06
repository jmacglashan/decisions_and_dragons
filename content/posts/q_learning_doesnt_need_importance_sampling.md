+++
title = "If Q-learning is off-policy, why doesn't it require importance sampling?"
date = 2024-04-02T23:43:37-04:00
draft = true
+++

In off-policy learning, we are evaluate the value function for a policy other than the one we are following in the environment. This difference creates a mismatch in state-action distributions. To account for this difference many actor-critic methods use importance sampling. However, Q-learning, perhaps the most famous off-policy method that evaluates the optimal policy, does not. There is a simple reason for that: In Q-learning, samples are only used to average over the transition dynamics, not the policy. Still confused? Let's make that more concrete with a simple example and derive different ways to solve it.<!--more-->

## Backseat gambling
Importance sampling is a method for estimating an expected value under one distribution using samples drawn from a different distribution. Classically, importance sampling was used deliberately because it turns out you can get a better estimate of an expected value using a different distribution than the one you care about[^1]. However, the main thing we're interesting in is that it let's you make estimates of expected values of one distribution with samples from another.

[^1]: Being able to get a better estimate using an alternative distribution turns out to be quite useful for estimating integrals, but that's another topic.

To turn that into a concrete but problem that is more related to reinforcement learning, let's imagine we're looking at two slot machines, which are more typically referred to as "bandits" in RL literature. Each bandit has a different payout rate that we don't know in advance. Our friend Alice told us that she is going to play from each 50-50. We watch her play for some time and observe her payouts for each pull. Later, our other friend Bob comes up and says he's going to play from from the left bandit 10% of the time and the right one 90% of the time. Let's call Alice's 50-50 strategy $\eta$ and Bob's biased strategy $\pi$. We denote their probabilities of picking the left or right as $\eta(l) = \eta(r) = \frac{1}{2}$ and $\pi(l) = \frac{1}{10}, \pi(r) = \frac{9}{10}$, respectively.

Can we use our observations of Alice to predict Bob's average payout? Yes, and there are multiple ways to do this (what a surprise, there are multiple ways to do this in RL too!). 

## Separate averages
The first way that you probably are thinking of solving this question is to compute the averge payout Alice receives on the left bandit and separately compute average payout Alice receives on the right. Let's denote these averages by 
$$
\begin{align*}
Q(l) = \frac{1}{N_l} \sum_i^{N_l} r_{li}, \\\
Q(r) = \frac{1}{N_r} \sum_i^{N_r} r_{ri}.
\end{align*}
$$
where $N_l, N_r$ are the number of times the left and right bandit were played, respectively and $r_{li}, r_{ri}$ are the ith payout for the left and right bandit, respectively.  

Not sure why I chose the letter $Q$. Just felt right.

After computing our averages, we can estimate[^3] Bob's expected payout by just taking a weight average of those values, where the weights are the probability Bob woud pick each bandit:
$$
E_{\pi}\left[R \right] \approx \pi(l) Q(l) + \pi(r) Q(r)
$$
And now we have our prediction for Bob's expected payout!

[^3]: This is expression approximately equals the expected payout because our Q-values are sample averages of the bandit payouts, not the true averages.

### An iterative way to estimate expected values
In the above examples, we estimated the expected value by taking the arithmetic mean of our samples. There is, however, an iterative way to estimate an expected value from samples. In the alternative way, we keep a current estimate of the expected value that can be arbitrarily initialized to any value and with each observation, we move a small step size toward the value of the new sample. We would formalize this update rule as:
$$
Q_{t+1}(i) \gets Q_t(i) + \alpha_t (r_t - Q_t(i))   
$$
where $r_t$ is the payout we observe at step $t$ and $\alpha_t \in (0, 1)$ is our learning rate at time step $t$. As long as $\alpha_t$ is slowly decreased with time under some mild constraints, this approach will converge to the true expected value.

In RL, and ML more broadly, this style of estimating expected values is often preferred because it works more clearly with function approximation and has some other nice properties like being smoother change to early on and allowing you to choose close values to initialize to if you have some background knowledge of the range of values.

At this point it should be clear to that if you simplified Q-learning to the bandit setting where there are no states, this iterative approach to estimating separate expected values for each arm/action is what Q-learning would become. And from it, we could compute the expected value for any policy by simply taking the weighted sum of the Q-values for the policy we're evaluating.

## Importance sampling
In importance sampling, we do not have to maintain separate averages for each choice. Instead, we are just going to modify our average over our data with some weights. Spoiler, we correct it by weighting them by $\frac{\pi(i)}{\eta(i)}$ where $i$ is either $l$ or $r$, depending on which bandit arm was pulled.

To show how we determine those weights let's start with how we would exactly compute Bob's expected payout. We'd ideally compute
$$
E_{\pi} \left[ R \right] = \sum_{i \in \\{l, r\\}} \pi(i) \sum_{r\in R} p_i(r) r,
$$
where $\pi_i(r)$ is the probability bandit $i$ will payout $r$.[^2] That is, we have a joint distribution over how Bob chooses and the payout distribution of each bandit. But there's just one problem: while we know the values of $\pi(l)$ and $\pi(r)$, we do not know what the values for $p_i(r)$ are. We only get to observe samples of pulls from each bandit.

[^2]: We're going to assume there are a discrete number of possible payouts, but what we're going to do would work for continuous values if we swapped the sum for an integral

If we were directly observing Bob play, this wouldn't be so bad. We would estimate the expected value with an average of his observed payouts. We wouldn't even have to keep track of which arm he pulled, because we can think of his random choice of the bandit as part of the underlying random distribution. But we don't get to observe Bob play. We're trying to predict his payout from observation of Alice.

But now comes the counter-intuitie importance sampling trick. Let's modify that expression without changing its value and then do some algebra:
$$
\begin{align*}
\sum_{i \in \\{l, r\\}} \pi(i) \sum_{r\in R} p_i(r) r &= \sum_{i \in \\{l, r\\}} \frac{\eta(i)}{\eta(i)} \pi(i) \sum_{r\in R} p_i(r) r \\\
&= \sum_{i \in \\{l, r\\}} \eta(i) \frac{\pi(i)}{\eta(i)} \sum_{r\in R} p_i(r) r \\\
&= \sum_{i \in \\{l, r\\}} \eta(i) \sum_{r\in R} p_i(r) \frac{\pi(i)}{\eta(i)} r \\\ 
&= E_{i \sim \eta} \left [ \sum_{r\in R} p_i(r) \frac{\pi(i)}{\eta(i)} r \right] \\\
&= E_{i \sim \eta} \left [ E_{r \sim p} \left[ \frac{\pi(i)}{\eta(i)} r \right] \right] \\\
\end{align*}
$$
Ahah! Now we turned our expected value from being of Bob's distribution $\pi$ into an expected value of Alice's distribution $\eta$! We just had to weight our payout values $r$ by $\frac{\pi(i)}{\eta(i)}$. Now we're back to a expected values of a joint distribution and we can approximate it with samples:
$$
E_{i \sim \eta} \left [ E_{r \sim p} \left[ \frac{\pi(i)}{\eta(i)} r \right] \right] 
    \approx \frac{1}{N} \sum_i^N \frac{\pi(a_i)}{\eta(a_i)} r_i 
$$
where $N$ is the total number of arm pulls Alice made between both bandits, $a_i$ indicates the bandit arm she pulled on the $ith$ try (either the left or right one), and $r_i$ is the payout she received.

Armed with this expression, we can approximate Bob's expected payout using our observations of Alice playing.

## Generalizing to MDPs
We now understand how Q-learning for simple bandits does not require importance sampling because we maintain separate expected value estimates for each action and can estimate the expected value of a different policy by taking a weighted average of the Q-values under that policy. We also understand that if we didn't keep seprate estimates for each action that importance sampling would be useful to correct for the mismatch is sampling distributions.

But what about more general MDPs where there are states and each action can lead you to another state? The main difference is to estimate our Q-values for these sequential MDPs, we have to account for the expected value of the policy we're evaluating in future states. But we now know how to solve that exact problem. Since Q-learning keeps separate estimates for each action, we can use the take a weighted average of future Q-values under the policy we care about (for Q-learning, the optimal policy) and never have to worry about importance sampling.

To illustrate that, let's work our way up from basics. Recall that the recursive definition of the optimal Q-function is
$$
Q^\*(s, a) = R(s, a) + \gamma \sum_{s'} T(s' | s, a) \max_{a'} Q^\*(s', a').
$$
Let's pause and ask ourselves: "where in this expression is the policy we're evaluting being used?" It's all the way to the right:
$\max_{a'} Q^\*(s', a')$ is the part of the expression that is defining the policy we're evaluating. That might not look like a policy to you because we're just taking a max of values. To remedy that, let's write down the Q-function for any arbitrary policy $\pi$, not just the optimal policy:
$$
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s'} T(s' | s, a) \sum_{a'} \pi(a' | s') Q^\pi(s', a')
$$
Now it's obvious in that expression. Also observe that we can write express optimal policy as a "stochastic" policy that with probability 1 selects the action with the highest Q-value[^5]:
$$
\pi^\*(a | s) = \begin{cases}
1 & \mathrm{if\ } Q^\*(s, a) = \max_{a'}Q^\*(s, a') \\\
0 & \mathrm{otherwise}
\end{cases}
$$
With that in mind, it follows that we can simplify the more general expression $\sum_{a'} \pi^\*(a' | s') Q^\*(s', a')$ to $\max_{a'} Q^\*(s', a')$.

Now we see that our recursive definition of the Q-funciton includes an expectation over the policy we're evaluating and it's estimating it using the weighted average of the Q-function that we've become familiar with.

## Q-learing: importance sampling free
Things should be getting clearer, but a recursive definition is not an algorithm and you might wonder if something is going to trip us up as move to algorithms for estimating Q. It won't, but let's be sure of that.

The recursive definition of the Q-function is neat on its own, but there is a really cool property that if you replace $Q^\*(s, a)$ with estimates and iteratively update the estimates using the recursive relationship, the values will converge to $Q^\*$! That is, using the update rule
$$
Q_{t+1}(s, a) \gets R(s, a) + \gamma \sum_{s'} T(s' | s, a) \max_{a'} Q_t(s', a')
$$
for all states and actions, we have that $Q_t \rightarrow Q^\*$ as $t \rightarrow \infty$. This algorithm is know as value iteration. In this post, I won't show why VI works, but let's simply accept that awesome property as true for the time being.

Of course, we wanted to know about Q-learning, not value iteraiton with Q-funciton. However, Q-learning is a lot like value iteration, except Q-learning is meant to solve *reinforcement learning* problems. In RL, the agent doesn't know the function $R(s, a)$ nor $T(s' | s, a)$, much like how in our bandit example we didn't know the probability distributions of payouts of the bandits $p$. But the agent can interact with the environment and obsereve the outcomes.

To handle this lack of knowledge of $R$ and $T$, Q-learning uses the iterative method for estimating expected values from samples:
$$
Q_{t+1}(s, a) \gets Q_t(s_t, a_t) + \alpha (r_t + \gamma \max_{a'} Q_t(s_{t+1}, a') - Q_t(s_t, a_t))
$$
Note that the samples are drawn from the reward and transition function, *not* the policy we're evaluating. In the one place we have to compute something about the policy: $\max_{a'} Q_t(s_{t+1}, a')$, we are using the seprate averages approach we talked about in our bandit example and using the simplificaiton of the expected value being equivalent to the max when evaluating the optimal policy. That is, the Q-function gives us a seperate expected value estimate for each action (in each state) and when we want to evaluate the expected average for some arbitrary policy (in this case, the optimal one), we just take the weighted combination of them.

[^5]: Or if there are ties for the highest Q-value, an optimal policy is any division of the probability between the actions that tie for the hightest value.

It should now be clear that even once we moved to Q-learning: a sample based algorithm that we didn't need importane sampling. It didn't, because in the part of the algorithm where we need to compute the expected value of our policy, we used the separate averages approach and did not neeed to correct for samples taken from a different policy.

## Why off-policy actor critic methods use importance sampling

After all this focus on Q-learning you may now actually be confused by off-policy actor critic algorithms need importane sampling! Not all actually do. It ultimately depends on whether the actor critic algorithm is estimating Q-functions, which have separate estimates for each action, or state value functions, which do not. 

To briefly review, one way we can define the state value function is in terms of $Q$:
$$
V^\pi(s) = \sum_a \pi(a | s) Q(s, a)
$$
If we had estimated $Q(s, a)$, we could have used the same approach of taking the weighted average. But many actor-critic algorithms do not. Some examples off-policy actor critic algorithms that do estimate Q and use this approach in SAC, TD3, DDPG, and all their subsequent variants. 

However, other off-policy actor critic algorithms like IMPALA.[^6] Instead, they directly use sampled returns to estimate $V$. Consequently, we're in the boat where are samples are drawn from a policy, but if it's a different policy that the polciy we're trying to evaluate. As such, we need to correct for the sample distribution difference with importance sampling, much like we did in our bandit example.

[^6]: IMPALA doesn't actually use the standard importance sampling correction ration we described above to mitigate the problem of compounding probabilities. They use an alternative, but similar one, that tends to work better. However, you certainly can use the one we described in off-policy actor critic algorithms.

## Why wouldn't we always learn Q-values?
You might wonder why we don't always estimate Q-values if we want to do off-policy learning. Afterall, it was probably the simpler approach you first imagined when I described the simple bandit problem. It also has the nice property that you don't have to know what the probabilities of the behavior policy was (e.g., Alice in our bandit example). You only have to know the probabilities of the policy you want to evaluate.

However, there are still some nice things about using state value estimates. First, if your action space is very large, maintaining separate estimates for each action can become problematic. If you're using function approximation, you might try to avoid that problem by generalizing over the actions. That is in fact what SAC, TD3, and DDPG do. But if you're introducing funtion approximation across your actions, now you've opened the door for more biased estimates for each action. That bias isn't trivial -- very often if algorithms like SAC fall apart it's inherently linked to bias issues in the estimate for each action. For these reasons, estimating the state value function and using importance sampling may be preferable.