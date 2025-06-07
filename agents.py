import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import pygame
import sys

class Agent(ABC):
    """Abstract base class for all reinforcement learning agents."""
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def select_action(self, state, training=True):
        """Select an action based on the current policy or value function."""
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update the agent's policy or value function."""
        pass

    @abstractmethod
    def train(self, episodes, display_game=False, plot_scores=True):
        """Train the agent over multiple episodes."""
        pass


class ValueBasedLearner(Agent):
    """Base class for value-based RL agents like Q-Learning and SARSA."""
    def __init__(self, game_env, learning_rate, discount_factor, exploration_rate,
                 epsilon_decay_rate, min_epsilon):
        super().__init__(game_env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate 
        self.initial_exploration_rate = exploration_rate 
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon

        self.action_space = list(range(self.env.action_space_size))
        
        self.q_table = {}  

    def select_action(self, state, training=True):
        if state not in self.q_table: 
            self.q_table[state] = {action: 0.0 for action in self.action_space}

        # Explore
        if training and np.random.rand() < self.exploration_rate:
            # return a random action from self.action_space (the set of possible actions) to ensure exploration
            return np.random.choice(self.action_space)
        # Exploit
        else:
            q_values = self.q_table[state]

            max_q = -float('inf')
            # Find the highest Q-value among all actions (store in max_q variable)
            for action_q in q_values.values():
                if action_q > max_q:
                    max_q = action_q
            
            # Collecting all actions that share the highest Q-value
            best_actions = [action for action, q_val in q_values.items() if q_val == max_q]
            
            if not best_actions: 
                return np.random.choice(self.action_space)
            return np.random.choice(best_actions) 

    def adjust_exploration_rate(self):
        # Reduce the exploration rate (epsilon) after each episode to gradually favor exploitation over exploration.
        # The new epsilon is the maximum of the minimum epsilon (min_epsilon) and the current epsilon multiplied
        # by the decay rate (epsilon_decay_rate), ensuring epsilon never drops below min_epsilon.
        self.exploration_rate = max(self.min_epsilon, self.exploration_rate * self.epsilon_decay_rate)

    def train(self, episodes, display_game=False, plot_scores=True):
        train_scores = []
        max_score_overall = -float('inf')
        print(f"Starting training for {episodes} episodes...")
        print(f"Initial Epsilon: {self.exploration_rate}, Min Epsilon: {self.min_epsilon}, Decay: {self.epsilon_decay_rate}")

        # Loop for each step of episode
        for episode in tqdm(range(1, episodes + 1), desc="Training Progress"):
            # Initializing S (state)
            state = self.env.reset()
            done = False
            current_episode_score = 0
            
            # Loop for each step of episode (until S is terminal)
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        print("Training interrupted by user. Exiting.")
                        sys.exit()
                if display_game:
                    self.env.render()
                    pygame.time.wait(max(1, int(1000 / self.env.speed)))

                # Choose A (action) from S (state) using policy derived from Q (explore or Exploit)
                action = self.select_action(state, training=True) 
                
                # Take A (action), observe R (reward), S' (next_state)
                next_state, reward, done = self.env.step(action)
                
                # Updating q_table
                self.update(state, action, reward, next_state, done)

                state = next_state
                current_episode_score += reward
            
            # Updating exploration rate (epsilon) to balance exploration and exploitation
            self.adjust_exploration_rate()

            train_scores.append(current_episode_score)
            if current_episode_score > max_score_overall:
                max_score_overall = current_episode_score
            if episode % 100 == 0:
                tqdm.write(f"Ep: {episode}/{episodes} - Score: {current_episode_score:.2f}, MaxScore: {max_score_overall:.2f}, Eps: {self.exploration_rate:.3f}")

        if plot_scores:
            print(f"\nTraining complete. States discovered: {len(self.q_table)}")
            self.plot_train_rewards(train_scores)
        return self.q_table

    def plot_train_rewards(self, train_scores):
        plt.figure(figsize=(12, 6))
        raw_episodes = np.arange(1, len(train_scores) + 1)
        plt.plot(raw_episodes, train_scores, label="Raw Score per Episode", color='grey', alpha=0.5)

        smoothing_window = min(50, len(train_scores) // 2 if len(train_scores) > 1 else 1)
        if smoothing_window > 0 and len(train_scores) >= smoothing_window:
            smoothed_scores = np.convolve(train_scores, np.ones(smoothing_window)/smoothing_window, mode='valid')
            smoothed_episodes = np.arange(smoothing_window, len(train_scores) + 1)
            plt.plot(smoothed_episodes, smoothed_scores, label=f"Smoothed Score (window={smoothing_window})", color='blue', linewidth=2)
        
        plt.xlabel("Episodes")
        plt.ylabel("Score")
        plt.title("Training Scores Over Episodes")
        plt.legend()
        plt.grid(True, linestyle="--")
        plt.tight_layout()
        plt.show()

    def play_final_game(self):
        print("\nInitiating final game with learned policy...")
        
        original_exploration_rate_backup = self.exploration_rate
        self.exploration_rate = 0.0
        print(f"Agent exploration rate temporarily set to: {self.exploration_rate}")

        menu_options_message = "Game finished. Close window or re-run script."
        self.env.final_show(self, menu_options_message) 

        self.exploration_rate = original_exploration_rate_backup # Restore
        print(f"Agent exploration rate restored to: {self.exploration_rate:.3f}")


class QLearningAgent(ValueBasedLearner):
    def __init__(self, game_env, learning_rate, discount_factor, exploration_rate,
                 epsilon_decay_rate, min_epsilon):
        super().__init__(game_env, learning_rate, discount_factor, exploration_rate,
                         epsilon_decay_rate, min_epsilon)

    def update(self, state, action, reward, next_state, done):
        # Initialize Q-values for the current state and next_state if not already present
        if state not in self.q_table:
            self.q_table[state] = {act: 0.0 for act in self.action_space}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {act: 0.0 for act in self.action_space}

        # Get the current Q-value for the (state, action) pair (store in current_q)
        current_q = self.q_table[state][action]
        
        # Calculate the TD error:
        # If next_state is terminal (done), future reward is 0
        # Otherwise, add the discounted max Q-value of next_state
        td_target = reward + self.discount_factor * (0 if done else max(self.q_table[next_state].values()))
        td_error = td_target - current_q

        # Update the state-action Q value using the learning_rate, current_q and TD error
        self.q_table[state][action] = current_q + self.learning_rate * td_error


class SARSAAgent(ValueBasedLearner):
    def __init__(self, game_env, learning_rate, discount_factor, exploration_rate,
                 epsilon_decay_rate, min_epsilon):
        super().__init__(game_env, learning_rate, discount_factor, exploration_rate,
                         epsilon_decay_rate, min_epsilon)

    def update(self, state, action, reward, next_state, done):
        # Initialize Q-values for the state and next_state if not already present
        if state not in self.q_table:
            self.q_table[state] = {act: 0.0 for act in self.action_space}

        if next_state not in self.q_table:
            self.q_table[next_state] = {act: 0.0 for act in self.action_space}

        # Select next action for SARSA update (by calling a function)
        next_action = self.select_action(next_state, training=True) if not done else None
        
        # Update Q-table
        # Calculate TD target using the reward plus discounted Q-value of the next_state and the next_action actually taken
        current_q = self.q_table[state][action]
        td_target = reward + self.discount_factor * (0 if done else self.q_table[next_state][next_action])
        self.q_table[state][action] = current_q + self.learning_rate * (td_target - current_q)