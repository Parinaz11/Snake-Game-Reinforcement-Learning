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
            # Initialize eligibility traces for SARSA(λ) if applicable
            if hasattr(self, 'eligibility_traces') and state not in self.eligibility_traces:
                self.eligibility_traces[state] = {action: 0.0 for action in self.action_space}

        # Explore
        if training and np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)
        # Exploit
        else:
            q_values = self.q_table[state]
            max_q = -float('inf')
            for action_q in q_values.values():
                if action_q > max_q:
                    max_q = action_q
            best_actions = [action for action, q_val in q_values.items() if q_val == max_q]
            if not best_actions: 
                return np.random.choice(self.action_space)
            return np.random.choice(best_actions) 

    def adjust_exploration_rate(self):
        self.exploration_rate = max(self.min_epsilon, self.exploration_rate * self.epsilon_decay_rate)

    def train(self, episodes, display_game=False, plot_scores=True):
        train_scores = []
        max_score_overall = -float('inf')
        print(f"Starting training for {episodes} episodes...")
        print(f"Initial Epsilon: {self.exploration_rate}, Min Epsilon: {self.min_epsilon}, Decay: {self.epsilon_decay_rate}")

        for episode in tqdm(range(1, episodes + 1), desc="Training Progress"):
            state = self.env.reset()
            done = False
            current_episode_score = 0
            
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        print("Training interrupted by user. Exiting.")
                        sys.exit()
                if display_game:
                    self.env.render()
                    pygame.time.wait(max(1, int(1000 / self.env.speed)))

                action = self.select_action(state, training=True) 
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                current_episode_score += reward
            
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
        self.exploration_rate = original_exploration_rate_backup
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


class MonteCarloAgent(ValueBasedLearner):
    def __init__(self, game_env, learning_rate, discount_factor, exploration_rate,
                 epsilon_decay_rate, min_epsilon):
        super().__init__(game_env, learning_rate, discount_factor, exploration_rate,
                         epsilon_decay_rate, min_epsilon)
        self.episode_data = []  # Store (state, action, reward) for each step in an episode
        self.returns = {}  # Track returns for each (state, action) pair
        self.visit_counts = {}  # Track number of visits for each (state, action) pair

    def update(self, state, action, reward, next_state, done):
        # Store the step data for Monte Carlo update at the end of the episode
        self.episode_data.append((state, action, reward))

    def train(self, episodes, display_game=False, plot_scores=True):
        train_scores = []
        max_score_overall = -float('inf')
        print(f"Starting Monte Carlo training for {episodes} episodes...")
        print(f"Initial Epsilon: {self.exploration_rate}, Min Epsilon: {self.min_epsilon}, Decay: {self.epsilon_decay_rate}")

        for episode in tqdm(range(1, episodes + 1), desc="Training Progress"):
            self.episode_data = []  # Reset episode data
            state = self.env.reset()
            done = False
            current_episode_score = 0

            # Collect episode data
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        print("Training interrupted by user. Exiting.")
                        sys.exit()
                if display_game:
                    self.env.render()
                    pygame.time.wait(max(1, int(1000 / self.env.speed)))

                action = self.select_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                current_episode_score += reward

            # Update Q-values using Monte Carlo returns
            G = 0  # Cumulative return
            for t in range(len(self.episode_data) - 1, -1, -1):
                state, action, reward = self.episode_data[t]
                G = reward + self.discount_factor * G  # Calculate return

                # Initialize returns and visit counts if not present
                if state not in self.returns:
                    self.returns[state] = {act: [] for act in self.action_space}
                    self.visit_counts[state] = {act: 0 for act in self.action_space}

                # Append return and increment visit count
                self.returns[state][action].append(G)
                self.visit_counts[state][action] += 1

                # Update Q-value using incremental mean
                if state not in self.q_table:
                    self.q_table[state] = {act: 0.0 for act in self.action_space}
                current_q = self.q_table[state][action]
                self.q_table[state][action] = current_q + (1 / self.visit_counts[state][action]) * (G - current_q)

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


class SARSALambdaAgent(ValueBasedLearner):
    def __init__(self, game_env, learning_rate, discount_factor, exploration_rate,
                 epsilon_decay_rate, min_epsilon, lambda_value=0.9):
        super().__init__(game_env, learning_rate, discount_factor, exploration_rate,
                         epsilon_decay_rate, min_epsilon)
        self.lambda_value = lambda_value  # Trace decay parameter
        self.eligibility_traces = {}  # Eligibility traces for state-action pairs

    def update(self, state, action, reward, next_state, done):
        # Initialize Q-values and eligibility traces for the state and next_state
        if state not in self.q_table:
            self.q_table[state] = {act: 0.0 for act in self.action_space}
            self.eligibility_traces[state] = {act: 0.0 for act in self.action_space}
        if next_state not in self.q_table:
            self.q_table[next_state] = {act: 0.0 for act in self.action_space}
            self.eligibility_traces[next_state] = {act: 0.0 for act in self.action_space}

        # Select next action for SARSA update
        next_action = self.select_action(next_state, training=True) if not done else None

        # Calculate TD error
        current_q = self.q_table[state][action]
        td_target = reward + self.discount_factor * (0 if done else self.q_table[next_state][next_action])
        td_error = td_target - current_q

        # Update eligibility trace for current state-action pair
        self.eligibility_traces[state][action] += 1.0

        # Update Q-values for all state-action pairs using eligibility traces
        for s in self.q_table:
            for a in self.q_table[s]:
                self.q_table[s][a] += self.learning_rate * td_error * self.eligibility_traces[s][a]
                self.eligibility_traces[s][a] *= self.discount_factor * self.lambda_value


class GLIEMonteCarloAgent(ValueBasedLearner):
    def __init__(self, game_env, learning_rate, discount_factor, exploration_rate,
                 epsilon_decay_rate, min_epsilon):
        super().__init__(game_env, learning_rate, discount_factor, exploration_rate,
                         epsilon_decay_rate, min_epsilon)
        self.episode_data = []  # Store (state, action, reward) for each step
        self.returns = {}  # Track returns for each (state, action) pair
        self.visit_counts = {}  # Track visit counts for each (state, action) pair

    def adjust_exploration_rate(self, episode):
        # GLIE exploration: epsilon = 1 / (episode + 1) for guaranteed exploration decay
        self.exploration_rate = max(self.min_epsilon, 1.0 / (episode + 1))

    def update(self, state, action, reward, next_state, done):
        # Store step data for Monte Carlo update
        self.episode_data.append((state, action, reward))

    def train(self, episodes, display_game=False, plot_scores=True):
        train_scores = []
        max_score_overall = -float('inf')
        print(f"Starting GLIE Monte Carlo training for {episodes} episodes...")
        print(f"Initial Epsilon: {self.exploration_rate}, Min Epsilon: {self.min_epsilon}")

        for episode in tqdm(range(1, episodes + 1), desc="Training Progress"):
            self.episode_data = []  # Reset episode data
            state = self.env.reset()
            done = False
            current_episode_score = 0

            # Collect episode data
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        print("Training interrupted by user. Exiting.")
                        sys.exit()
                if display_game:
                    self.env.render()
                    pygame.time.wait(max(1, int(1000 / self.env.speed)))

                action = self.select_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                current_episode_score += reward

            # Update Q-values using Monte Carlo returns
            G = 0  # Cumulative return
            for t in range(len(self.episode_data) - 1, -1, -1):
                state, action, reward = self.episode_data[t]
                G = reward + self.discount_factor * G

                if state not in self.returns:
                    self.returns[state] = {act: [] for act in self.action_space}
                    self.visit_counts[state] = {act: 0 for act in self.action_space}

                self.returns[state][action].append(G)
                self.visit_counts[state][action] += 1

                if state not in self.q_table:
                    self.q_table[state] = {act: 0.0 for act in self.action_space}
                current_q = self.q_table[state][action]
                self.q_table[state][action] = current_q + (1 / self.visit_counts[state][action]) * (G - current_q)

            self.adjust_exploration_rate(episode)  # GLIE-specific decay
            train_scores.append(current_episode_score)
            if current_episode_score > max_score_overall:
                max_score_overall = current_episode_score
            if episode % 100 == 0:
                tqdm.write(f"Ep: {episode}/{episodes} - Score: {current_episode_score:.2f}, MaxScore: {max_score_overall:.2f}, Eps: {self.exploration_rate:.3f}")

        if plot_scores:
            print(f"\nTraining complete. States discovered: {len(self.q_table)}")
            self.plot_train_rewards(train_scores)
        return self.q_table