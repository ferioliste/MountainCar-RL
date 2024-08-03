# Dealing with sparse rewards in the Mountain Car environment

This repository contains an implementation of different reinforcement learning algorithms to solve the Mountain Car problem. This problem is a classic example in the field of reinforcement learning where the goal is to drive an underpowered car up a steep hill. An agent must learn to leverage potential energy of the car by oscillating back and forth to build enough momentum to reach the top of the hill on the right. In particular, the reward at each step is set at -1 except when the goal is reached where it is 0. The main challenge of this environment is the sparse reward function.

The code was developed for the course "Artificial neural networks/reinforcement learning" at EPFL in the academic year 2023-2024, spring semester.

Check out the final report here: [`report_MountainCar_PaoloGiaretta_StefanoFerioli.pdf`](./report_MountainCar_PaoloGiaretta_StefanoFerioli.pdf).

## Authors
- Stefano Ferioli ([@ferioliste](https://github.com/ferioliste))
- Paolo Giaretta ([@GiarettaPaolo](https://github.com/GiarettaPaolo))

## Implemented agents
This project implements mainly two agents: Deep Q-Network (DQN) and Dyna.

DQN agent uses a feed-forward neural network to approximate the Q-values structure in each state. The agent uses an $\epsilon$-greedy policy for action selection, balancing exploration and exploitation.
**Features:** Replay buffer, batch learning, and an epsilon decay schedule to reduce the exploration rate over time.
**Tested reward structures:**
1. environment reward
2. environment reward + heuristic reward (obtained as mechanic energy variation between consecutive states)
3. environment reward + RND reward (obtained through Random Network Distillation [[Burda et al., 2018]](https://arxiv.org/pdf/1810.12894) to encourage exploration of less frequently encountered states, the advantage of this type of reward is that it is non domain-specific)

Dyna agent uses a model-based reinforcement learning approach to create a model of the dynamics of the enviroment. The agent discretizes the state space and integrates real experiences with random updates to learn its policy.
**Features:** State discretization, transition probability modeling, and reward estimation for actions given the state.

## Repository structure
- `agents/` contains the impelemntation of the agents as well as some auxiliary functions
- `notebooks/` contains the Jupyter notebooks used to independently test the agents. The data collected dutring the test is automatically saved in `results/` and the generated plots are saved in `plots/`. Saving the generated data in files allows to generate the plots without having to run each test again.
- `notebook_MountainCar_PaoloGiaretta_StefanoFerioli.ipynb` collects in one notebook all the notebooks in `notebooks/`. Running the whole file allows to automatically generate most of the plots presented in the report.