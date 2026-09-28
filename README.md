# Lunar Lander Agents

A project focused on developing and comparing two autonomous agents for the **LunarLander** environment:

* **Reactive Agent**: makes decisions based only on the current state of the environment.
* **Evolutionary Agent**: uses an evolutionary approach to optimize the agent's behaviour through successive generations.

## Project Overview

The goal of the project is to explore two different approaches to autonomous decision-making and evaluate their performance in the LunarLander environment.

The agents are developed independently, allowing their behaviour, learning process, and performance to be compared under the same environment and evaluation criteria.

## Agents

### Reactive Agent

The reactive agent selects an action based directly on the current state of the environment.

It does not maintain a long-term memory or learn from previous episodes. Its behaviour is determined by a set of rules that map the current state to an action.

### Evolutionary Agent

The evolutionary agent uses an evolutionary algorithm to optimize its decision-making strategy.

A population of candidate solutions is evaluated over multiple generations. Better-performing solutions are selected and modified through evolutionary operators such as mutation and, where applicable, crossover.