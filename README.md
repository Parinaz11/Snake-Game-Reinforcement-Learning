# Reinforcement Learning Snake Game
The project aims to explore and implement various model-free RL algorithms, specifically Monte Carlo (MC) and Greedy in the Limit with Infinite Exploration (GLIE) Monte Carlo, to develop an optimal playing policy for the Snake agent.

## The Snake Game Environment
The Snake game serves as the environment for our RL agent. In this game:

- The player controls a "snake" that moves around a grid.

- The objective is to eat "food" (represented as an apple) that appears randomly on the grid.

- Each time the snake eats food, it grows longer, and the player's score increases.

- The game ends if the snake collides with the boundaries of the grid or with its own body.

The challenge for an RL agent is to learn a policy that maximizes the total reward (score) over time by efficiently navigating the grid, consuming food, and avoiding self-collision or wall collision.

![snake_game_image](https://github.com/user-attachments/assets/93a04818-fb3e-4eb8-b2ed-8f4745e6a465)

Figure 1 - an image of the Snake game

## Core Problem: Learning Optimal Policies with Model-Free RL
The central problem addressed in this project is to implement and analyze model-free reinforcement learning algorithms to enable an AI agent to play the Snake game effectively. Specifically, the focus was on:

### Implementing Every-Visit Monte Carlo (MC) Control
The first phase involves implementing the Every-Visit Monte Carlo algorithm. This algorithm learns the optimal action-value function Q(s,a) by averaging returns from episodes. Key aspects of this phase include:

- Data Collection: Gathering state, action, and reward triplets for each step within an episode. Unlike Temporal Difference (TD) methods (like SARSA and Q-learning), Q-values are not updated immediately at each step but rather after the completion of an entire episode.

- Return Calculation: After an episode concludes, calculating the discounted cumulative return (G) for each step by working backward from the end of the episode.

- Q-value Update: Updating Q(s,a) values using an incremental mean approach, considering every occurrence of a state-action pair within an episode.

- Exploration Strategy: Employing an ϵ-greedy exploration strategy with a fixed ϵ value throughout the algorithm.

### Implementing GLIE Monte Carlo (GLIE MC) Control
The second phase extends the learning by implementing the Greedy in the Limit with Infinite Exploration (GLIE) Monte Carlo algorithm. This algorithm builds upon the standard Monte Carlo approach but incorporates a decaying exploration rate to ensure infinite exploration of all state-action pairs in the long run, while gradually converging towards a greedy policy.

Key Components of GLIE MC:

- Episode Data Collection: Similar to standard MC, the agent interacts with the environment, selects actions based on an ϵ-greedy policy, and stores sequences of states, actions, and rewards.

- Return Calculation: Calculating the discounted return (G) for each step after an episode.

- Q-value Update: Updating Q(s,a) using the incremental mean of observed G values, weighted by the number of visits to that pair.

- GLIE Exploration Rate: The exploration rate ϵ decays according to the formula, which ensures sufficient exploration while the policy eventually becomes greedy, satisfying the GLIE condition.

**Note:** Like Monte Carlo and GLIE, Q-Learning and SARSA are also implemented as model-free algorithms.

## Getting Started
To run this project locally, follow these steps:

### Prerequisites

- Python 3.x
- `pygame` library

You can install `pygame` using pip:
```bash
pip install pygame
```
To execute the program, run:

```bash
python main.py
```

Upon execution, you will be prompted to select the desired algorithm. After the algorithm completes, a reward plot (similar to Figure 2) will be displayed, illustrating the agent's progressive improvement in scores throughout training.

![Reward_Plot_MC](https://github.com/user-attachments/assets/b238f21c-2201-471e-ad48-95f632cde487)

Figure 2 - Reward plot for MC

