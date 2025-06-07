import pygame
import pandas as pd

from graphics.snake_env import SnakeEnv
from agents import QLearningAgent, SARSAAgent

num_episodes = 5000
display_game_during_training = False 
plot_training_scores = True
show_final_game_after_training = True

discount_factor = 0.99 # gamma
learning_rate = 0.1
initial_exploration_rate = 1.0
epsilon_decay_rate = 0.999
min_epsilon = 0.01

pygame.init() 

environment = SnakeEnv()
environment.speed = 30

# --- Q-learning algorithm ---
learner = QLearningAgent(
    game_env=environment,
    learning_rate=learning_rate,
    discount_factor=discount_factor,
    exploration_rate=initial_exploration_rate,
    epsilon_decay_rate=epsilon_decay_rate,
    min_epsilon=min_epsilon
)

# --- SARSA algorithm ---
# learner = SARSAAgent(
#     game_env=environment,
#     learning_rate=learning_rate,
#     discount_factor=discount_factor,
#     exploration_rate=initial_exploration_rate,
#     epsilon_decay_rate=epsilon_decay_rate,
#     min_epsilon=min_epsilon
# )

q_table = learner.train(
    episodes=num_episodes,
    display_game=display_game_during_training,
    plot_scores=plot_training_scores
)

if q_table is None:
    print("Training returned None. Check agent's train method.")
elif not q_table:
    print("Q-table is empty after training. This might indicate an issue.")
else:
    print(f"\nQ-table has {len(q_table)} states.")
    q_table_data = []
    for state, action_values in q_table.items():
        for action, value in action_values.items():
            q_table_data.append((str(state), action, value))
    q_table_df = pd.DataFrame(q_table_data, columns=["State", "Action", "Value"])
    print(f"Sample of Q-table DataFrame:\n{q_table_df.head()}")
    q_table_df.to_csv("q_table.csv", index=False)
    print("Q-table saved to q_table.csv")


if show_final_game_after_training:
    learner.play_final_game()

pygame.quit() 
print("\nProgram finished.")