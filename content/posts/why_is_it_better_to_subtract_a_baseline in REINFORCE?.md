+++
title = 'Why is it better to subtract a baseline in REINFORCE?'
date = 2025-05-08T15:06:56-04:00
draft = false
+++

Suppose we have a softmax policy over five actions. In the current state, an oracle tells us that the true Q-values are -1, -2, -3, -4, and -5. The first action is the best, even though it is negative, so we should increase the probability of it. We sample an action from our policy. It's the first! What a stroke of luck!

Or is it? Unfortunately, our algorithm is REINFORCE. It decreases the probability of selecting the winning action. Why? Because the REINFORCE stochastic gradient is the product of the Q-value estimate and the gradient of the log probability of the action: $\nabla \log \pi(a | s) Q^\pi(s, a)$. Since the Q-value is negative, REINFORCE decreases the action's probability.

Unless we're using a baseline. For example, if we subtracted a baseline of -3, then we would multiply the gradient of the log probability by $(-1 - -3) = 2$: a positive weight. In this case, REINFORCE correctly increases the probability of the action.
<!--more-->

## What's the point of REINFORCE again?

After the above example, you might be wondering why REINFORCE is so fickle. Why do we need to bother with baselines to make REINFORCE behave? Why not just increase the probability of the best action to begin with?! That's because in our example, we cheated: an oracle told us the Q-values for every action. Not only that, there were only five actions, so it was easy to determine how we should improve the policy.

But what if an oracle didn't tell us what all the Q-values were? Or what if there were so many actions that we couldn't exhaustively search for the best? What if all we got to observe was a Q-value estimate for the single action we tried in the environment?

Suddenly, determining how to improve our policy isn't so obvious. Irritatingly, this is exactly the situation RL presents us. It's actually quite amazing that REINFORCE works at all.

To better understand why baselines make a big difference, let's review the relationship between REINFORCE's stochastic gradients and the true policy gradient. If you'd like to see a full derivation of REINFORCE's stochastic gradient, see my answer [here](../why_does_the_policy_gradient_include_log_prob/).


At any state $s$, the true policy gradient ($g^*$) that we want to compute is:
$$
g^\* = \sum_a \nabla_\theta \pi(a | s)Q^\pi(s, a)
$$
For optimizing the policy objective, we would want to average these gradients over the (discounted) state distribution, much like how in supervised learning we average the gradients of the loss for each instance in a dataset. We can ignore that for simplicity.

The difficulty with the true gradient is it requires summing the gradient of the policy times its Q-value for *each* action. That can be computationally intractable to do from each state. Worse, it is impossible if we're using empirical returns to estimate Q-values because we only observe the return for the action we took in the environment.

REINFORCE[^2] solves these problems by deriving a stochastic gradient $g_a$ for any single action $a$ sampled from $\pi$:

$$
g_a = \nabla_\theta \log \pi(a | s) Q^\pi(s, a)
$$

REINFORCE's stochastic gradient estimate has the property that
$$
E_{a \sim \pi(\cdot | s)} \left[g_a \right] = g^\* .
$$

Therefore, as long as we keep generating samples, we can use stochastic gradient ascent[^1] to average the REINFORCE gradients and move in the direction of the true gradient.

Nevertheless, as we saw in the intro, any single sample can still point in the wrong direction, and this error can make our optimization challenging. Ideally, we'd like each stochastic gradient to be as close to the true gradient as possible.

[^2]: It could be argued that it's not REINFORCE when you use the Q-values, and is only REINFORCE if you are averaging over empirical returns. Naturally, the average over empirical returns is the Q-value which is why that works. However, the strict original definition of a REINFORCE algorithm [by Williams](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) is one that uses the empirical reward signal or return. It was the subsequent [Policy Gradient paper](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) that used the Q-function framing. Methods that use anything beyond the empirical return and a subtracted baseline are more properly "actor critic" methods. E.g., if you use an n-step TD lambda value, generalized advantage estimation, or a learned Q-function estimate, then you're using an actor critic method, not REINFORCE. Nevertheless, this detail is unimportant to the mathematical machinery we're discussing.

[^1]: If you multiply the whole expression by negative one, then you can use stochastic gradient *descent* instead of ascent.


## The problem with stochastic gradients

To illustrate how bad stochastic gradients can be, let's compute some concrete gradients for our example problem.

We'll first need to define what the logits of the softmax policy are. We'll assume they're zero everywhere, yielding a uniform probability distribution. Normally, the logits would be a function of the state, like a neural net. If a neural net was predicting those logits as a function of the state, the gradient w.r.t. the logits would be backpropagated to the rest of the neural net to effect the change. For simplicity's sake, we can restrict our focus to the gradient w.r.t. the logits. The rest is just arbitrary architecture choices.

### The true policy gradient
With the help of JAX, let's compute the *true* policy gradient at this state.

```python
import jax
import jax.numpy as jnp
import jax.nn as nn


def policy_objective(logits, qs):
    return jnp.sum(nn.softmax(logits) * qs)


policy_grad = jax.grad(policy_objective)


logits = jnp.zeros(5)
qs = jnp.array([-1.0, -2.0, -3.0, -4.0, -5.0])

print("True policy gradient", policy_grad(logits, qs))
```

Rounding, the output is: `[0.4 0.2 0.0 -0.2 -0.4]`

Stepping in the direction of the true gradient increases the logit for the best action, as we would expect. It also increases the logit for the second best action, though not quite as much. The third action logit is unchanged, and the the worst two action logits are decreased.

### REINFORCE gradients
Now let's compute the stochastic gradient REINFORCE would estimate for each action.

```python
def reinforce_sample_obj(logits, qs, action):
    log_probs = nn.log_softmax(logits)
    return log_probs[action] * qs[action]


reinforce_sample_grad = jax.grad(reinforce_sample_obj)


def reinforce_grads(logits, qs):
    return jnp.stack(
	    [
		    reinforce_sample_grad(logits, qs, i)
		    for i in range(logits.shape[0]) # grad for each action
		],
		axis=0
	)

rgrads = reinforce_grads(logits, qs)
print(rgrads)
```

The output is
```python
[[-0.8  0.2  0.2  0.2  0.2]  # grad when we sample a1
 [ 0.4 -1.6  0.4  0.4  0.4]  # grad when we sample a2
 [ 0.6  0.6 -2.4  0.6  0.6]  # grad when we sample a3
 [ 0.8  0.8  0.8 -3.2  0.8]  # grad when we sample a4
 [ 1.   1.   1.   1.  -4. ]] # grad when we sample a5
```

If we sample the first action -- the best action -- REINFORCE would decrease the logit for it, while increasing the logits for the other actions. Consequently, the probability for taking the best action will be decreased, while the probability for the suboptimal actions will be increased. That's certainly not what we want!

However, we also observe that REINFORCE would decrease the probability for whichever action we sample, because the Q-value is always negative. Furthermore, sampled suboptimal actions will have their probability more greatly decreased than the best action. On average, REINFORCE will increase the probability of the best action!

Let's prove that to ourselves by computing the expected value of the REINFORCE gradients.

```python
probs = nn.softmax(logits).reshape((5, 1)
print(jnp.sum(probs * rgrads, axis=0))
```

As expected, we recover the true policy gradient we computed earlier: `[0.4  0.2 0.0 -0.2 -0.4]`.

Therefore, as long as we generate enough samples and use small enough step sizes, REINFORCE will move us in the direction of the true gradient. It sure would be nice if we could reduce the error so that we didn't worry so much about the number of samples and step size though.

### Gradient error

If we want to reduce the error, we'll first need a way to quantify how bad the error between a sample gradient and the true gradient is. Let's define the error between a REINFORCE gradient at action $a$ ($g_a$) and the true policy gradient ($g^*$) to be their squared error:
$$
e_a \triangleq \left(g_a - g^\* \right)^2
$$
Our gradients $g_a$ and $g^\*$ are vectors, so we can compute this error element-wise, making $e_a$ a vector as well.

Great, that's a sensible definition to quantify the error. However, it would be nice to summarize this error across all the stochastic gradients. To do that, let's pick the expected value of this error:
$$
e \triangleq E_{a\sim \pi(\cdot | s)}\left[e_a \right]
$$
You may have noticed that we just defined $e$ to be the (element-wise) variance of the REINFORCE gradients because $g^\*$ is the expected value of the REINFORCE gradients!

When you hear people say that REINFORCE is a high-variance estimator, you should translate that to meaning that there is a high average squared error between the REINFORCE gradients and the true policy gradient. That can make optimization difficult.

To demonstrate that, let's compute the average error of the REINFORCE gradients in our example.

```python
def reinforce_error(logits, qs):
    rgrads = reinforce_grads(logits, qs)
    probs = nn.softmax(logits).reshape((-1, 1))
    true_grad = policy_grad(logits, qs)
    pvar = jnp.sum(probs * jnp.square(rgrads - true_grad), axis=0)
    return pvar

print(reinforce_error(logits, qs))
```

Rounding, the output is `[0.4 0.88 1.52 2.32 3.28]`. Indeed, this is a non-trivial amount of error!


## Baselines reduce error
Earlier, we built some intuition about why REINFORCE gradients can be so bad. If all values are negative and we sample the best action, REINFORCE will decrease its probability. That suggests that the error (variance) may be reduced if we subtract a baseline $b$ from the Q-value:
$$
g_a^b = \nabla_\theta \log \pi(a | s) (Q^\pi(s, a) - b)
$$
so that the weight for the best action $(Q^\pi(s, a_1) - b)$ is positive instead of negative.

Let's try subtracting a baseline of -2 and see what the error is. That will be enough to make the best action have a positive weight: $-1 - -2 = 1$

```python
print(reinforce_error(logits, qs - -2))
```

The output is approximately `[0.08 0.08 0.24 0.56 1.04]`.

Indeed, this has reduced the error quite a bit!

### ...Or make it worse
Of course, we have to be a little careful. If the baseline is too extreme, it can make the error even worse than it was without a baseline.

For example, let's try over-correcting with a baseline of -10. That choice will still make the best action have a positive weight, but all the suboptimal actions will now have positive weights too.

```python
print(reinforce_error(logits, qs - -10))
```

We find the error with a baseline of -10 is indeed even worse than no baseline: `[11.6 9.68 7.92 6.3 4.88]`

Just like all weights being negative has the undesirable effect that REINFORCE will decrease the probability for any single action sample (including the best action), all weights being positive has the undesirable effect that REINFORCE will increase the probability for any single action sample (including the worst actions). Like before, the average gradient will still equal the true gradient, but the error of individual stochastic gradients can be quite high.

## How to choose baselines

So if subtracting a baseline can make the error better or worse, how should we choose the baseline?

You might be tempted to analytically find the optimal baseline. And indeed, you can work out the math for that! But I have some bad news: the solution will require summing (or integrating) over all actions and their Q-values. If we could do that, we wouldn't need to use REINFORCE to begin with!

Fortunately, there is a good choice that is not as prohibitive as computing the optimal baseline.
### Just use $V^\pi(s)$
I'm sure you saw it coming that setting the baseline to the state value $b \triangleq V^\pi(s)$ hits the sweet spot for reducing the error and being easy to estimate/compute. It's the defacto choice for many RL algorithms, after all. But perhaps now we're a bit better equipped to understand why.

What we'd really like is to increase the probability of any action that does better than our policy on average, and to decrease the probability of any action that does worse than our policy on average. If we could do that exactly, then every update would result in a policy improvement, regardless of what sample we drew!

The state value function is precisely what measures the average value under the current policy. So if we subtract it from our Q-values, we'll achieve the desired property!

There is one catch to that -- if you constrain your policy to a parametric distribution, you might not always get a policy improvement. For example, if you are using a Gaussian distribution over continuous actions, moving the mean parameter will increase the probability of nearby actions. And if those nearby actions are very bad (think of walking along a cliff), then the policy may regress!

However, you need perverse situations for that to be a significant problem. Furthermore, it remains a problem whether you subtract a baseline or not. So on the whole, you're much better off using $V^\pi(s)$ as a baseline!

To convince ourselves, let's try using $V^\pi(s)$ as the baseline and check the error. (Note: this will make the baseline `-3`.)
```python
print(reinforce_error(logits, qs - policy_objective(logits, qs)))
```

Rounding, the output is: `[0.4, 0.16, 0.08, 0.16, 0.4]`.

The error for the best action and second best action is higher than it was when we used -2, but the error is substantially reduced for the worst actions than when we used -2. If we summed up all the element-wise errors, we'd find that $b = V^\pi(s) = -3$ has a total error of only $1.2$, while our previous choice of $b = -2$ had a total error of $2$. So indeed, this heuristic worked out pretty well!

### What about  $V^\pi$ estimation errors?
Naturally, you don't have $V^\pi(s)$ handed to you. You can only estimate it in different ways. For example, by using a separate neural network trained with some flavor of TD. If your estimate for $V^\pi(s)$ is poor, it won't help as much. Nevertheless, it's still probably better than doing nothing!

### What about $Q^\pi$ estimation errors?
These can bite you too! Whether you're using empirical returns from a trajectory (which is what makes an algorithm a "proper" REINFORCE algorithm[^2]), TD-$\lambda$ returns, or a Q-function neural net, all of these have sources of variance or error. Baseline subtraction sadly won't help you with these kinds of errors. It only helps you with the error/variance inherent to stochastic REINFORCE gradients.

## Why not learn a Q-function and avoid this problem entirely?

A common question at this point is why not just learn a Q-function and select the best action? Sure, it's not the oracle ground truth we used in our example, but if it's good enough we could search it for the best action and avoid all this funny REINFORCE and baseline business.

Maybe you can. That's certainly what Q-learning does! Unfortunately, that has two problems that may not make it an appropriate solution. The first problem is a learned Q-function introduces bias into your policy improvement. The nice thing about empirical returns is that they are an unbiased estimate.

The second problem is you have to make an insidious assumption: that you can easily search for the best action. If you only have a handful of actions, it's not hard to search for the action with the highest Q-value using your learned Q-function. What if you have many billions? That's sadly the situation we run into with multidimensional 'continuous' actions. (No, continuous actions does not translate to infinite values to search because computers can only represent finite values with high precision. But the number is, uh, big.) And even if you have purely discrete actions, the number of unique actions can still grow quite quickly if you have multiple discrete action dimensions.

If there are too many actions, learning a Q-function isn't good enough. We need some way to efficiently search for good actions or estimate a policy gradient. REINFORCE avoids that problem by sampling actions and computing the stochastic REINFORCE gradient.
