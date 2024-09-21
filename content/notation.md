+++
title = 'Math Notation Cheatsheet'
date = 2024-03-26T20:17:59-04:00
draft = false
type = "page"
layout = "single_no_toc"
+++

In this section I outline the meaning of the mathematical notation I use. When appropriate (and possible), I also describe the meaning in simple python.

## General math and statistics

#### $f(x) \triangleq mx + b$

The $\triangleq$ indicates that the expresson on the left is _defined_ to be the expression on the right, rather than an equivalence that is derived from mathemtical rules.

#### $(a, b)$

The set of real numbers between $a$ and $b$, excluding those values.

#### $[a, b]$

The set of real numbers between $a$ and $b$, including those values.

#### $(a, b]$ and $[a, b)$

The set of real numbers between $a$ and $b$, excluding the left or right bound, respectively.

#### $\\{ a, b, c \\}$

A set containing elements a, b, and c.

#### $x \in X$

Indicates that value $x$ is an element in set $X$.

#### $x_i$

The $ith$ element of the indexed set (list) of elements $x$. In python `x[i]`.

#### $\sum_i^N x_i$

The sum of the first $N$ elements of an indexed set (list) of elements $x$. Could also be written as $x_1, + x_2 + ... + x_N$.

```python
sum = 0
for i in range(N)
    sum += x[i]
```

Whether the starting index is 0 or 1 depends on the context of the variable. If $i$ is assigned to a value like $\sum_{i=3}^N$, it means the series starts at element $i$. If it is unclear whether the starting index is 0 or 1, it will sometimes be explicitly assigned.

#### $\sum_{x \in X} x$

The sum of the elements in $X$.

```python
sum = 0
for x in X:
    sum += x
```

#### $\prod_i^N x_i$

The product of the first $N$ elements of an indexed set (list) of elements $x$. Could also be written as $x_1 \cdot ... \cdot x_N$

```python
prod = 1
for i in range(N):
    prod *= x[i]
```

Whether the starting index is 0 or 1 depends on the context of the variable. If $i$ is assigned to a value like $\prod_{i=3}^N$, it means the series starts at element $i$. If it is unclear whether the starting index is 0 or 1, it will sometimes be explicitly assigned.

#### $x_{t+1} \gets g(x_t)$

The $\gets$ arrow indicates that the value of $x$ is updated to be the result of some funciton/operation on the previous value of $x$ defined on the right-hand-side (it doesn't have the be $g$ and can be any expression). The subscript $t$ indicates the value of $x$ after the
$t$th update to it.

#### $E_p[X]$

The expected value of random variable $X$ drawn from probability distribution $p$. For discrete random variables this is defined as $E_p[X] \triangleq \sum_{x \in X} p(x) x$. For continuous random variables this is the integral $E_p[X] \triangleq \int_X p(x) x dx$.

```python
expected_x = 0.0
for x in X:
    expected_x += p(x) * x
```

#### $x \sim p$

Random variable $x$ is drawn from probability distribution $p$.

#### $E_{x \sim p} [ g(x) ]$

The expected value of drawing $x$ from probability distribution $p$ and then applying some operation on it -- in this case evaluating the function $g$ on it. For discrete random variables $E_{x \sim p} [ g(x) ] \triangleq \sum_{x \in X} p(x) g(x)$.

#### $p(x | y)$

The probabiltiy (or probability density) of $x$ given the value $y$ from conditional probability (mass/density) function $p$

#### $p( \cdot | y)$

The conditional probability distribution (rather than specific probability/density) conditioned on $y$ implied by the probability (mass/density) function $p$.

## RL-specific variables and notation choices

#### $S$

State space

#### $A$

Action space

#### R(s, a) or R(s, a, s')

A reward function on state-action pairs or state-action-next state triples. In practice, reward functions are usually defined in terms of the state $s' \in S$, while theoretical analysis usually assumes the simpler $R(s, a)$ functions. Doing analysis with $R(s, a)$ does not limit the validity of the analysis because for any MDP with an $R(s, a, s')$ function, you can define an equivalent MDP with an $R(s, a)$ function that is the expected value of the $R(s, a, s') values (averaging over the next state probabilities).

#### T(s' | s, a)

The proability (density) that the environment transitions to state $s' \in S$ after the agent takes action $a \in A$ from state $s \in S$.

#### $\gamma$

A geometric discount factor $\gamma \in [0, 1)$.

#### $\pi(s)$

A deterministic policy that maps state $s \in S$ to an action.

#### $\pi(a | s)$

The probability (mass/density) of stochastic policy $\pi$ selecting action $a \in A$ from state $s \in S$.

#### $\pi(\cdot | s)$

A stochastic policy distribution conditioned on state $s$.
