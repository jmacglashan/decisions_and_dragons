+++
title = 'Why is the DDPG gradient the product of the Q-function gradient and policy gradient?'
date = 2024-04-20T12:51:21-04:00
draft = true
+++

The DDPG and DPG paper before it express the gradient of the objective $J(\pi)$ as the product of the policy and Q-function gradients:

$$
\nabla_\theta J(\pi) = E_{s \sim \rho^\pi} \left[\nabla_\theta \pi_\theta(s) \nabla_a Q(s, a) \rvert_{a \triangleq \pi_\theta(s)} \right].
$$

This expression looks a little scary, but it's conveying a straightforward concept: the gradient is the average of the Q-function's gradient with respect to the policy parameters, evaluated at the policy's selected action. That may not be obvious because the product of "gradients" (spoiler: there is some notation abuse) is the result of applying the multivariable chain rule of differentiation. If we were to reverse this step, the expected value would simplify to the more explicit expression $\nabla_\theta Q(s, \pi_\theta(s))$.

Let's rederive this result and explore why the literature presents it with an expansion of the chain rule instead of the more recognizable form.<!--more-->

## Revisting policy iteration

Like so many methods in RL, the foundation of DDPG and DPG is policy iteration. Policy iteration is a template algorithm for improving your policy. It is described by two steps that you repat until the policy stops improving:

1. Evaulate $Q^\pi(s, a)$ for your current policy $\pi$ for all state-action pairs.
2. Improve your policy $\pi$ by updating it to maximize $Q(s, \pi_\theta(s))$ for each state.

This algorithm templates makes sense intutively. We want our policy to maximize the expected future discounted reward. The Q-function $Q^\pi(s, a)$ tells us how much expected future discounted reward the agent will receive if it takes take action $a$ from state $s$ and then follows policy $\pi$ thereafter. Therefore, by updating $\pi$ to maximize $Q(s, \pi(s))$ at each state, the agent will improve its return!

Many different concrete algorithms emerge from how we choose to do steps 1 and 2. For the purposes of this topic, we'll focus on step 2. When an MDP has a small discrete set of actions and a finite managable set of states, you can implement step 2 by enumerating the Q-values for each action in each state and updating the policy to select the action with the highest Q-value.

Unfortunately, if the MDP has continuous actions and a very large or continuous state space, this approach isn't going to work. DDPG and DPG address this problem by using gradient descent.

## Q-functions as loss functions

When actions are continuous and the states are large, we cannot find the action with the highest Q-value by simply enumerating each action's value. We also cannot directly assign that action to be selected in each state if the state space is too large or continuous. However, optimizing a neural net polciy to maximize the Q-function should look very similar to training a neural net.

### Supervised regression

Suppose we have a supervised regression problem where we have a dataset of $N$ $(x_i, y_i)$ pairs where $x_i$ is some input vector and $y_i$ is a real value we want to predict from x. We train a neural net $f_\theta$ with parameters $\theta$ by first defining a loss funciton and minimizing the the composition of the loss function and our network predictions via (stochastic) gradient descent.

For example, typically we first define the loss function to be the squared error between a label $y$ and some predicted value $\hat{y}$:

$$
L_2(y, \hat{y}) \triangleq \frac{1}{2}(y - \hat{y})^2
$$

The closer this loss is to zero, the better the prediction $\hat{y}$ is.

Then, given our differentiable neural net $f_\theta$, we define a loss function _for the network_ as the average of a composition of our loss $L_2$ and our network predicitons, for each $(x_i, y_i)$ pair in our dataset:

$$
L(\theta) \triangleq \frac{1}{N} \sum_i^N L_2(y_i, f_\theta(x_i))
$$

By computing the gradient of this loss, we can minimize the loss using (stochastic) gradient descent:

$$
\begin{align*}
\theta_{t+1} & \gets \theta_t - \alpha \nabla_{\theta_t} L(\theta_t) \\\
&= \theta_t - \alpha \nabla_{\theta_t} \frac{1}{N} \sum_i L_2(y_i, f_{\theta_t}(x_i)) \\\
&= \theta_t - \alpha \frac{1}{N} \sum_i \nabla_{\theta_t} L_2(y_i, f_{\theta_t}(x_i)) \\\
&= \theta_t - \alpha E_{i} \left[ \nabla_{\theta_t} L_2(y_i, f_{\theta_t}(x_i)) \right]
\end{align*}
$$

where $\alpha$ is a learning rate.

### Policy improvement

When we lay out regular supervised neural net training, we see it looks a lot like our Q-function maximization problem. Here are some minor differenes we can easily bridge.

1. We don't have an indexed dataset, we have a distribution over states.
2. We don't have labels, but the Q-function does depend on the state.
3. We want to maximize the Q-function, not minimize a loss.
4. Our policy output is possbly mutli-dimensional.

The first is easily addressed by chagning our distribution in the expected value. The second just means we don't input a label, we input a state. The third is addressed by turning our Q-function into a loss to minimize by multiplying by negative one. The fourth only matters implicitly to how gradients are computed, which we will return to in a moment. Making those substitutions, we get:

$$
\theta_{t+1} \gets \theta_t - \alpha E_{s\sim \rho^\pi} \left[ -\nabla_{\theta_t} Q(s, \pi_{\theta_t}(s)) \right]
$$

The take away is that you can think of the negative Q-function as a loss function for training a neural net policy. By minimizing this loss with (stochastic) gradient descent, we solve the policy improvement problem for MDPs with continuous actions and large or continuous state spaces.[^1]

[^1]: **Warning**: A crucial limitation of stochastic gradient descent (SGD) is that it only converges to a local optimum. While this limitation is often manageable in supervised learning, it can be a significant issue in policy improvement. That is, suprevised learning problems typically involve a convex loss function with a single global optima, like $L_2$. When the function approximation architecture is also convex (like a linear function), SGD will converge to the global optima. While neural networks are not convex, bad local optima can be avoided by just making the neural net bigger. However, in DDPG, both the network and the Q-function, which serves as our loss function, may be non-convex. Because this problem is inherent to the loss function, simply scaling up the network will not resolve it. Scaling up the Q-function network won't help either, because we're training the Q-function network to model the true Q-function, and the true Q-funtion may not be convex.

But we're still not quite at the expression in the DDPG and DPG literature that is the product of gradients. Let's work our way back to that.

## Revisiting the chain rule

Our next question is how to go from $\nabla_\theta Q(s, \pi_\theta(s))$ to $\nabla_\theta \pi_\theta(s) \nabla_a Q(s, a) \rvert_{a \triangleq \pi_\theta(s)}$. This final step is just a result of applying the multivariable chain rule.

To explain, let's briefly review the single variable chain rule of differentiation. Given a function $h$ that is defined to be a composition of function $f$ and $g$: $h(x) = f(g(x))$, we can compute the derivative of $h$ as product of the derivative of $f$ and the deriviative $g$:

$$
h'(x) = f'(g(x))g'(x).
$$

The chain rule is useful because it allows us to simplify the computation of derivatives of complex functions by decomposing the function into sub functions for which we already know the derivative. E.g., if I asked you to compute the derivative of $\sin(x^2)$, you don't have to do the exhaustive work of solving a limit of the expression. Instead, if you already know the derivative of $\sin(x)$ and the derivative of $x^2$, then you can use the chain rule to compute the derivate of $\sin(x^2)$.

### Multivariable functions

Of course, that's just single variables. Once we have multivariable functions, as is common in the case of neural networks, it get's a little more complex. But fortunately, not wildly more complex, because there is also a [chain rule for multivariable funcitons](https://en.wikipedia.org/wiki/Gradient#Chain_rule)!

Suppose our function $f$ is a function of multiple variables (like a Q-function of multi-dimensional continuous actions), suppose $g$ outputs a multi-dimensional value in the same domain as $f$ (like a deterministic continuous-action policy), and suppose our input $x$ for which we want the gradient is multidimensional (like the parameters $\theta$ of a neural net). Then the multivariable chain rule is:

$$
\nabla_x h(x) = (Dg(x))^\top \nabla_a f(a) \rvert_{a=g(x)},
$$

where $(Dg(x))^\top$ indicates the transpose of the Jacobian matrix (the matrix of gradients for each output) and $\rvert_{a=g(x)}$ indicates that we should evaluate the value $a$ in $f(a)$ as the value of $g(x)$.

The multivariable chain rule is useful for the same reason as the single-variable chain rule: it allows you to leverage your knowledge of the gradients of simple functions to compute the gradient of a complex function that is a composition of those functions.

### Applying the chain rule

Applying the chain rule to our DDPG/DPG gradient, we have:

$$
\begin{align*}
\nabla_\theta Q(s, \pi_\theta(s)) = (D\pi_\theta(s))^\top \nabla_a Q(s, a) \rvert_{a=\pi_\theta(s)},
\end{align*}
$$

where in this case the (transposed) Jocobian $(D\pi_\theta(s))^\top$ has its column gradients be with respect to the underlying parameters $\theta$ (we are treating state $s$ as a constant since we are optimizing the parameters, not states!).

That looks almost identical to what we see in literature, except instead of $D\pi_\theta(s))^\top$, the literature writes $\nabla_\theta \pi_\theta(s)$. I regret to inform you that this difference is just an abuse of notation. If you look at description after equation 7 in the [DPG paper](https://proceedings.mlr.press/v32/silver14.pdf), you will see they simply define $\nabla_\theta \pi_\theta(s)$ to be the transpose Jacobian.

## Why explicitly expand the chain rule?

We've finally seen how to arrive at the DDPG/DPG policy gradient, but there is a good chance you have a final question: in an era of autodifferentiation, why does literature expand the chain rule? Why not just use the more recognizeable expression $\nabla_\theta Q(s, \pi_\theta(s))$? After all, with an autodifferentiation library we will simply compute $Q(s, \pi_\theta(s))$ and then ask the library to compute the gradient for us.

Although I can't know for certain what drove the authors to expand the chain rule, I can offer some plausible reasons behind their decision. Here are a few possible motivations:

**(1)** The first reason is historical. Although it's hard to imagine a time when we wern't all using autodifferentiation libraries, that time did exist, and it wasn't _that_ long ago. If you don't have automatic differentiation, you're going to want a clear expression that you would know how to implment yourself. By expanding the chain rule, it makes it clear that you need a way to compute the gradient of the Q-function with repsect to the actions, and the Jacobian of the policy with respect to the policy parameters.

Before deep networks took over, RL researchers often used simple linear functions. Linear functions have very simple gradients that you can compute efficiently. As a result, researchers could easily calculate the Jacobian and gradients of their linear policy and Q-functions separately and then compute the product, without relying on an automatic differentiation system. In fact, the DPG paper specifically focused on linear function approximation.

**(2)** The second reason is that in addition to deriving the deterministic policy gradient, the original DPG work also showed that this gradient is the limit of stochastic policies as they converge to deterministic policies. By using the expanded chain rule form, it may have been easier to prove this relationship with stochastic policies.

**(3)** The third reason is you need to be careful with automatic differentiation. If you compute the gradient of the Q-function with an autodiff library you need to take care not to compute the gradients for the Q-function parameters during the optimization. Otherwise, you may accidentally distort the Q-function network. By expanding the chain rule, it makes it clear exactly what should be computed. That said, there are lots of ways in modern automatic differentiation libraries to avoid this pitfall without expanding the chain rule, so I would not suggest expanding it in modern practice. Just be aware of it.

## Bias in the DDPG estimate

As a concluding remark, it is important to observe that the DDPG/DPG approach has a source of bias in it. Ideally, we'd be differentiating the true Q-function. But we almost never have this function. Instead, we train another neural net to estimate the Q-function and then take gradients of that. Because we use an estimate of the Q-function, the gradients of that esitmate, and the gradient of our policy objective in turn, will be biased.
