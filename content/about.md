+++
title = 'About'
date = 2024-03-26T20:17:59-04:00
draft = false
type = "page"
layout = "about"
+++

Reinforcement learning (RL) is an elegant problem definition for autonomous agents that learn from their own interactions with an environment. But the methods to solve this simple problem definition are not so simple. To solve this problem you must simultaneously tackle many subproblems that are all complex enough to warrant their own subfields in AI, such as perception, prediction, planning, and memory.

Furthermore, unlike other forms of machine learning, an RL algorithm is not provided well-curated datasets. An RL agent must form its own data from interactions. Even worse, the data the agent collects is highly correlated and does not explicitly include the correct response. Instead, an RL agent must reason about its data to determine the correct reponse and it must actively explore the environment to ensure it has good data coverage. If that wasn't hard enough for you, solving this problem often involves learning multiple interacting models.

The RL problem is _hard_ and if you feel lost trying to grock it, you're not alone.

This website aims to address that problem. Here I've collected answers I've given to common questions in the past and expanded on them.

Each answer begins with a concise response, followed by a step-by-step derivation from first principles. The aim of these step-by-step derivations is to fill in the gaps commonly skipped over in papers.

I can't promise that you'll come away from these answers with perfect clarity on the matter. The unpleasant reality is most of us have to suffer a long time to gain competence. But perhaps this site will help you suffer a little less.

## About the name

This site is called "Decisions & Dragons. "Decisions" represents the core goal of RL: developing agents that learn to make effective decisions in an environment. "Dragons" represents the perilous complexities and challenges that must be navigated in pursuit of solving the RL problem.

I trust you understand that there were no other motivations for this name and any similarities it has with other titles is purely coincidental.

## How I will update the site

At the time of launch, I populated this site with answers to frequently asked questions that I've encountered and addressed in the past. I've also taken this opportunity to expand on those previous answers to address follow up questions I received and to take advantage of the freedom of presentation this site affords.

Moving forward, I will continue to curate and share my responses to new and emerging questions, as well as revisit and refine my previous answers as needed.

Occasionally, I may use this platform to share opinion pieces on RL and AI more broadly, although I am less sure of this direction. This site is still a work in progress -- we'll see how it goes.

## About me

I'm James MacGlashan. If you want to ask me RL questions, the best place is either on [Twitter](https://twitter.com/jmac_ai), or
on the [Reinforcement Learning Discord server](https://discord.gg/nu3pyBrNpg). (See the other links at the top of this page.)

I received my PhD in computer science from the University of Maryland, Balitmore County in 2013 where I
worked on reinforcement learning. I then moved on to a postdoctoral position at Brown University, where I continued to work on reinforcement learning.
Following my postdoc, I joined the startup Cogitai, where we worked to build reinforcement learning and
continual learning as a service. Cogitai was eventually acquired by Sony and we formed game AI team at [Sony AI](https://ai.sony/), where I
continue to work on reinforcement learning.

Despite all these years working on reinforcement learning, I have shockingly failed to solve it.

Fortunately, it hasn't all been bad news for the field. RL methods have vastly improved
and I've played a role in bringining reinforcement learning to products with
[GT Sophy](https://www.gran-turismo.com/us/gran-turismo-sophy/) -- an RL agent that outraced the best racers in the game Gran Turismo Sport. GT Sophy was subsequently adapted to be a racing opponent in Gran Turismo 7 that you can race against today!

We're continuing to work on exciting reinforcement learning applications and problems at Sony AI and I hope we can help
turn reinforcement learning into a robust technology that can be more broadly used. Perhaps one that is less
fraught with dragons.
