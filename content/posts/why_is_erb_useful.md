+++
title = 'Is experience replay only useful for neural net training?'
date = 2024-05-26T15:06:56-04:00
draft = true
+++

Experience replay buffers (ERB) were popularized by [DQN](https://training.incf.org/sites/default/files/2023-05/Human-level%20control%20through%20deep%20reinforcement%20learning.pdf) as a means to de-correlate data -- a property critical for training deep neural networks. That fact suggests that ERBs are only useful for deep RL. However, the idea is much older and was originally used to propagate values through bootstrapping faster. This property makes ERBs useful for any bootstrapping-based RL method, from deep RL to tabular RL.

<!--more-->

## Faster value propagation

## De-correlating data
