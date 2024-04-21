+++
title = 'Why is the DDPG gradient the product of the Q-function gradient and policy gradient?'
date = 2024-04-20T12:51:21-04:00
draft = true
+++

The DDPG and DPG paper before it express the gradient of the objective $J(\pi)$ as the product of the policy and Q-function gradients:

$$
\nabla_\theta J(\pi) = E_{s \sim \rho^\pi} \left[\nabla_\theta \pi_\theta(s) \nabla_a Q(s, a) \rvert_{a \triangleq \pi_\theta(s)} \right].
$$

This expression looks a little scary, but all it's saying is to update your policy with gradient ascent to maximize the Q-function. This might not be obvious because the expression has been expanded with the multivariable chain rule of differentiation. If we undid the the chain rule, the product of "gradients" inside the expected value would be replaced with $\nabla_\theta Q(s, \pi_\theta(s))$.

Let's talk more about how we got here and why literature might express the result with the chain rule applied in the era of modern automatic differentiation.<!--more-->

## Revisting policy iteration

Like so many methods in RL, the foundation of DDPG is policy iteration. Policy iteration is a template algorithm for improving your policy. It is described by two steps that you repat until you stop getting a better policy:

1. Evaulate $Q^\pi(s, a)$ for your current policy $\pi$ for all state-action pairs.
2. Improve your policy $\pi$ by updating it to maximize $Q(s, \pi_\theta(s))$ for each state.

This algorithm templates makes intuitive sense. We want our policy to maximize the expected future discounted reward. The Q-function $Q^\pi_\theta(s, a)$ tells us how much expected future discounted reward we'll get if take action $a$ from state $s$ and then follow our policy $\pi$ afterwards. So if we update $\pi$ to maximize $Q(s, \pi_\theta(s))$ at each state, we'll get better return!

Lot's of different concrete algorithms emerge from how we choose to do steps 1 and 2. If you have a small discrete set of actions and you have a small enough finite set of states, you can implement step 2 by enumerating the Q-values for each action in each state and updating your policy to select the action with the highest value.

Unfortunately, if you have continuous actions and a very large or continuous state space, this approach isn't going to work. But we can use gradient descent to save us.

## Q-functions as loss functions

When actions are continuous and the states are large, we cannot enumerate the action that maximizes the Q-function and cannot update the policy to directly output that action. But this problem should look very similar to training a neural net.

That is, suppose we have a supervised regression problem where we have a dataset of $(x, y)$ pairs where $x$ is some input vector and $y$ is a real value we want to predict from x. We train a neural net $f_\theta$ with parameters $\theta$ by first defining a loss funciton and minimizing the the composition of the loss function and our network predictions via (stochastic) gradient descent.

For example, typically first we define the loss function as the squared error between a label $y$ and some predicted value $\hat{y}$:

$$
L_2(y, \hat{y}) \triangleq \frac{1}{2}(y - \hat{y})^2
$$

The closer this loss is to zero, the better the prediction $\hat{y}$ is.

Then, given our differentiable neural net $f_\theta$, we define a loss function _for the network_ as the average of a composition of our loss $L_2$ and our network predicitons for each label in our dataset:

$$
L(\theta) \triangleq \frac{1}{N} \sum_i^N L_2(y_i, f_\theta(x_i))
$$

By computing the gradient of this loss, we can use (stochastic) gradient descent to update our nework $f_\theta$ to minimize the loss:

$$
\begin{align*}
\theta & \gets \theta - \alpha \nabla_\theta L(\theta) \\\
 &= \theta - \alpha \nabla_\theta \frac{1}{N} \sum_i L_2(y_i, f_\theta(x_i)) \\\
 &= \theta - \alpha \frac{1}{N} \sum_i \nabla_\theta L_2(y_i, f_\theta(x_i)) \\\
 &= \theta - \alpha E_{i} \left[ \nabla_\theta L_2(y_i, f_\theta(x_i)) \right]
\end{align*}
$$

where $\alpha$ is a learning rate.

When we lay out regular supervised neural net training, we see it looks a lot like our Q-function maximization problem. Here are some minor differenes we can bridge easily.

1. We don't have an indexed dataset, we have a distribution over states.
2. We want to maximize the Q-function, not minimize a loss
3. Our policy output is possbly mutli-dimensional

The first is easily addressed by chagning our distribution. The second is easily addressed by turning our Q-function into a loss to minimize by multiplying by negative one. The third only matters implicitly to how gradients are computed, which we will return to in a moment. Making those simple substitutions, we get:

$$
\theta \gets \theta - \alpha E_{s\sim \rho^\pi} \left[ -\nabla_\theta Q(s, \pi_\theta(s)) \right]
$$

The take away is that you can think of the negative Q-function evaluated at your policy like a loss function for training a neural net. By minimizing this loss with (stochastic) gradient descent, we solve the policy improvement problem of how to maximize the Q-function on a space of continuous acitons.[^1]

[^1]: Beware: one limitation of stochastic gradient is it only finds local optimas. This is not often a problem in neural nets, but in this formulation it can be more insiduous. That is, in normal regression problems our loss function $L_2$ for any point has just one local optima, which is the global optima. What this means is solving just the surface of the loss for any input is easy. Any local optima problems we run into in supervised learning has to do with the neural net architecutre having local optimas in parameter space. As it turns out, if you just make your parameter space big (i.e., make a big neural net), it's hard to get stuck local optimas due to the neural net architecture. However, unlike our usual supervised loss $L_2$, the Q-function may have local optimas over its actions! And the action space is what it is, you cannot just make it bigger. Consequently, using SGD to optimize the Q-function is more prone to getting stuck in local optimas.

But we're still not quite at the expression in the DDPG and DPG literature that is the product of gradients. Let's work our way back to that.

## Revisiting the chain rule

Our next question is how to go from $\nabla_\theta Q(s, \pi_\theta(s))$ to $\nabla_\theta \pi_\theta(s) \nabla_a Q(s, a) \rvert_{a \triangleq \pi_\theta(s)}$. This final step is just a result of applying the multivariable chain rule to our gradient.

To explain, let's briefly review the single variable chain rule of differentiation. That is, given a function $h$ that is defined as a function $f$ composed of a funciton $g$: $h(x) = f(g(x))$, we can compute the derivative of $h$ as product of the derivative of $f$ and the deriviative $g$:

$$
h'(x) = f'(g(x))g'(x).
$$

The chain rule is useful because it allows us to simplify the computation of derivatives of complex functions by decomposing the function into sub functions for which we already know the derivative. E.g., if I asked you to compute the derivative of $\sin(x^2)$, you don't have to do the exhaustive work of solving a limit of the expression. Instead, if you already know the derivative of $\sin(x)$ and the derivative of $x^2$, then you can use the chain rule to compute the derivate of $\sin(x^2)$.

### Multivariable functions

Of course, that's just single variables. Once we have multivariable functions, as is common in the case of neural networks, it get's a little more complex. But fortunately, not wildly more complex, because there is a [chain for multivariable funcitons](https://en.wikipedia.org/wiki/Gradient#Chain_rule) too!

That is, suppose our function $f$ is a function of multiple variables (like a Q-function of multi-dimensional continuous actions), suppose $g$ outputs a multi-dimensional value in the same domain as $f$ (like a deterministic continuous-action policy) and suppose our input $x$ is also multidimensional (like the parameters $\theta$ we will update). Then the multivariable chain rule is:

$$
\nabla_x h(x) = (Dg(x))^\top \nabla_a f(a) \rvert_{a=g(x)},
$$

where $(Dg(x))^\top$ indicates the transpose of the Jacobian matrix (the matrix of gradients for each output) and $\rvert_{a=g(x)}$ indicates that we should evaluate the value $a$ in $f(a)$ as the value of $g(x)$.

This multivariable chain rule is useful for the same reason the single variable one: if you know the gradient/derivative for a bunch of functions, the chain rule lets you reuse that knowledge to compute the gradient of a funciton that is a composition of those functions with known gradients/derivatives.

### Applying the chain rule

Applying the chain rule to our expression, we have:

$$
\begin{align*}
\nabla_\theta Q(s, \pi_\theta(s)) = (D\pi_\theta(s))^\top \nabla_a Q(s, a) \rvert_{a=\pi_\theta(s)},
\end{align*}
$$

where in this case the (transposed) Jocobian $(D\pi_\theta(s))^\top$ has its column gradients be with respect to the underlying parameters $\theta$ (we are treating state $s$ as a constant since we are optimizing the parameters, not states!).

That looks almost identical to what we see in literature, except instead of $D\pi_\theta(s))^\top$, the literature writes $\nabla_\theta \pi_\theta(s)$. I regret to inform you that there is no further insight to this, it's just an abuse of notation. If you look at description after equation 7 in the [DPG paper](https://proceedings.mlr.press/v32/silver14.pdf), you will see they simply define $\nabla_\theta \pi_\theta(s)$ to be the transpose Jacobian.

## Why we explicitly express the chain rule

We've finally seen how to arrive at the DDPG/DPG policy gradient, but there is a good chance you final question: in an era of autodifferentiation, why does literature expand the chain rule to define the gradient? Why not just leave it at $\nabla_\theta Q(s \pi_\theta(s))$? After all, in an autodifferentiation we will simply compute $Q(s \pi_\theta(s))$ and then as the library to compute the gradient for us.

While I cannot tell you for certain what the exact motivations of the authors were for expanding the chain rule, I can give you a few reasons why they might.

**(1)** The first reason is historical. Although autodifferentiation is now a staple and it's hard to imagine a time when we wern't all using it, that time did exist, and it wasn't _that_ long ago. If you don't have automatic differentiation, you're going to want a clear expression that you would know how to implment yourself. By expanding the chain rule, it makes it clear that you need a way to compute the gradient of the Q-function with repsect to the actions, and the Jacobian of the policy with respect to the function parameters.

Before deep networks took over, simple forms of funtion approximation were often used in RL, such as simple linear functions. Linear functions have very easy gradients that you can compute efficiently, Therefore, even without an autodifferentiation system, it would be easy for practictioners to separately compute the gradients of their linear policy and Q-functions. Indeed, the original DPG paper focused on linear function approximation.

**(2)** The second reason is that part of the original DPG work was to not only present the deterministic policy gradient, but to show that this gradient is the limit of stochastic policies as they converge to deterministic policies. By taking this form, it may have been easier to show this limiting relationship with stochastic policies.

**(3)** The third and final reason is you do need to be a little careful with automatic differentiation. If you just take a gradient of the Q-function with autodiff and accidentally compute gradients for the Q-function parameters as well as the policy parameters, you're going to get incorret results! By expanding the chain rule, it helps make it clear exactly what you should be doing. That said, there are lots of ways in modern automatic differentiation libraries to avoid this pitfall without actually expanding the chain rule yourself.

## Bias in the DDPG estimate

As a final note, it's worth being aware that the DDPG/DPG approach has a source of bias in it. Ideally, we'd be differentiating the true Q-function. But we never have this function. Instead, we train another neural net to estimate the Q-function and then take gradients of that. Because we use an estimate of the Q-function, the gradients of that esitmate, and the gradient of our policy objectie in turn, will be biased.
