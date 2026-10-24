# Reinforcement Learning for Snake

<p align="center">
  <strong>Teaching an agent to play Snake through model-free reinforcement learning.</strong>
  <br>
  <sub>Explore Monte Carlo methods, GLIE exploration, Q-learning, and SARSA in a classic game environment.</sub>
  <br><br>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-238636?style=for-the-badge" alt="Reinforcement Learning">
  <img src="https://img.shields.io/badge/Pygame-6A5ACD?style=for-the-badge" alt="Pygame">
</p>

## Overview

This project explores how a reinforcement learning agent can learn to play Snake through interaction with a game environment. It implements and examines multiple **model-free reinforcement learning algorithms**, focusing on how agents learn action-value functions, balance exploration and exploitation, and improve their performance over repeated episodes.

The project covers Every-Visit Monte Carlo control and Greedy in the Limit with Infinite Exploration (GLIE) Monte Carlo control, alongside Q-learning and SARSA. Training rewards can be plotted to observe changes in the agent's performance over time.

## The Game Environment

<p align="center">
  <img src="https://github.com/user-attachments/assets/93a04818-fb3e-4eb8-b2ed-8f4745e6a465" alt="Snake game environment" width="480">
  <br>
  <sub>Figure 1. The Snake game environment.</sub>
</p>

The Snake game provides a simple environment for studying sequential decision-making.

- **Objective:** Guide the snake toward randomly appearing food.
- **Growing state:** Eating food increases the snake's length and score.
- **Movement constraints:** The snake moves on a grid and must avoid walls and its own body.
- **Episode termination:** A collision with a wall or the snake's body ends the game.

The agent's goal is to learn a policy that maximizes cumulative reward by collecting food while avoiding collisions.

You can also try a version of the game by searching for [Google Snake](https://www.google.com/search?q=Google+Snake) in your browser.

## Algorithms

<table>
  <tr>
    <td align="center" width="50%">
      <h3>Monte Carlo Control</h3>
      <img src="https://img.shields.io/badge/Every--Visit-Monte%20Carlo-3973AC?style=flat-square" alt="Every-Visit Monte Carlo">
      <p>Estimates action values from complete episodes and updates them using observed returns.</p>
    </td>
    <td align="center" width="50%">
      <h3>GLIE Monte Carlo</h3>
      <img src="https://img.shields.io/badge/GLIE-Decaying%20Exploration-8250DF?style=flat-square" alt="GLIE Monte Carlo">
      <p>Uses a decaying exploration rate to balance continued exploration with increasingly greedy action selection.</p>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <h3>Q-learning</h3>
      <img src="https://img.shields.io/badge/Model--Free-Off--Policy-238636?style=flat-square" alt="Q-learning">
      <p>Learns action values using a greedy target policy, independently of the action-selection policy.</p>
    </td>
    <td align="center" width="50%">
      <h3>SARSA</h3>
      <img src="https://img.shields.io/badge/Model--Free-On--Policy-D97706?style=flat-square" alt="SARSA">
      <p>Updates action values using the state, action, reward, next state, and next action experienced by the agent.</p>
    </td>
  </tr>
</table>

### 1. Every-Visit Monte Carlo Control

Monte Carlo methods learn from complete episodes rather than updating action values at every individual time step.

The implementation follows these steps:

- **Collect experience:** Store the state, action, and reward at each step.
- **Calculate returns:** After an episode ends, calculate discounted cumulative returns backward through the episode.
- **Update action values:** Update \(Q(s,a)\) using the observed returns and an incremental mean.
- **Explore:** Use an epsilon-greedy policy with a fixed exploration rate.

Every-Visit Monte Carlo considers every occurrence of a state-action pair within an episode when updating its value estimate.

### 2. GLIE Monte Carlo Control

GLIE stands for *Greedy in the Limit with Infinite Exploration*. It extends the Monte Carlo approach with an exploration strategy designed to preserve exploration while gradually favoring greedy actions.

The learning process includes:

- Collecting state, action, and reward sequences over complete episodes.
- Calculating discounted returns after each episode.
- Updating action values using incremental averages.
- Decaying the exploration rate over training.

A GLIE exploration schedule is intended to provide sufficient exploration while allowing the policy to become greedy in the limit. The exact behavior depends on the exploration schedule implemented in the code.

### 3. Q-learning and SARSA

The project also includes two temporal-difference methods:

| Property | Q-learning | SARSA |
|---|---|---|
| Learning approach | Temporal difference | Temporal difference |
| Policy type | Off-policy | On-policy |
| Update target | Greedy next-action value | Value of the next action actually selected |
| Experience | Updates from individual transitions | Updates from individual transitions |

Unlike Monte Carlo control, these methods update action values during interaction with the environment rather than waiting for an entire episode to finish.

## Learning Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/1-Initialize%20Agent-3973AC?style=flat-square" alt="Step 1">
  <br>↓<br>
  <img src="https://img.shields.io/badge/2-Observe%20Game%20State-3973AC?style=flat-square" alt="Step 2">
  <br>↓<br>
  <img src="https://img.shields.io/badge/3-Select%20Action-8250DF?style=flat-square" alt="Step 3">
  <br>↓<br>
  <img src="https://img.shields.io/badge/4-Interact%20with%20Environment-8250DF?style=flat-square" alt="Step 4">
  <br>↓<br>
  <img src="https://img.shields.io/badge/5-Record%20Reward%20and%20Next%20State-238636?style=flat-square" alt="Step 5">
  <br>↓<br>
  <img src="https://img.shields.io/badge/6-Update%20Policy%20or%20Action%20Values-238636?style=flat-square" alt="Step 6">
  <br>↓<br>
  <img src="https://img.shields.io/badge/7-Track%20Training%20Performance-D97706?style=flat-square" alt="Step 7">
</p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <strong>Episode continues</strong><br><br>
      ↓<br>
      Observe the next state<br><br>
      ↺ Select another action
    </td>
    <td align="center" width="50%">
      <strong>Episode ends</strong><br><br>
      ↓<br>
      Calculate episode returns for Monte Carlo<br><br>
      ↓<br>
      Update action values as required<br><br>
      ↓<br>
      Begin another episode
    </td>
  </tr>
</table>

The precise update timing differs by algorithm. Monte Carlo methods update from completed episodes, whereas Q-learning and SARSA update from individual transitions.

## Training Results

The project plots rewards to help visualize the agent's performance during training.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b238f21c-2201-471e-ad48-95f632cde487" alt="Monte Carlo training reward plot" width="560">
  <br>
  <sub>Figure 2. Reward plot for Monte Carlo training.</sub>
</p>

Reward plots provide a way to examine changes in episode scores over training. They can help identify performance trends, although individual rewards may fluctuate because of exploration and the stochastic elements of the environment.

## Technology Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pygame-00A86B?style=flat-square" alt="Pygame">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib">
</p>

- **Python 3** — Implementation of the environment and learning algorithms.
- **Pygame** — Game environment and graphical rendering.
- **Matplotlib** — Training reward visualization.

## Getting Started

### Prerequisites

- Python 3.x
- Pygame
- Any additional dependencies imported by the project

### 1. Clone the repository

```bash
git clone https://github.com/Parinaz11/SnakeGame_RL.git
cd SnakeGame_RL
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install pygame matplotlib numpy
```

Install any other packages required by the imports in the project.

### 4. Run the program

```bash
python main.py
```

Upon execution, the program prompts you to select an algorithm. After training completes, a reward plot is displayed to help you examine the agent's performance.

## Project Goals

- Implement and explore model-free reinforcement learning algorithms.
- Understand the difference between Monte Carlo and temporal-difference learning.
- Examine exploration strategies and the effect of epsilon-greedy action selection.
- Track training rewards and observe how agent performance changes over time.

## Acknowledgments

Inspired by the classic Snake game and developed as a practical exploration of reinforcement learning algorithms.
