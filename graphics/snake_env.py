import random
from collections import deque
import pygame
import os 
import sys

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SNAKE_COLOR = (108, 102, 227)
SNAKE_DARKER = (80, 75, 184)
DARKER_GREEN = (162, 208, 74)
LIGHTER_GREEN = (170, 216, 82)
GREEN_BUTTON = (165, 252, 119)
RED_BUTTON = (230, 80, 80)
DARK_GRAY = (46, 42, 42)

action_to_direction = {
    "UP":    {0: "UP",    1: "LEFT",  2: "RIGHT"},
    "DOWN":  {0: "DOWN",  1: "RIGHT", 2: "LEFT"},
    "LEFT":  {0: "LEFT",  1: "DOWN",  2: "UP"},
    "RIGHT": {0: "RIGHT", 1: "UP",    2: "DOWN"}
}

class SnakeEnv():
    def __init__(self):
        pygame.display.set_caption("Snake RL Environment")
        self.window_width = 440
        self.window_height = 440
        self.square_size = 20
        self.margin = 20
        self.speed = 20
        
        self.snake = deque()
        self.action_space = list(range(3))
        self.action_space_size = len(self.action_space)
        self.state_size = 5

        self.screen = None

        try:
            base_path = os.path.dirname(__file__)
            apple_path = os.path.join(base_path, "apple.png")
            self.apple_image = pygame.image.load(apple_path)
            self.apple_image = pygame.transform.scale(self.apple_image, (self.square_size, self.square_size))
        except pygame.error as e:
            print(f"Warning: Could not load apple.png. Pygame error: {e}")
            print("Using default red square for apple.")
            self.apple_image = None
        
        self.reset()

    def step(self, action):
        self.direction = action_to_direction[self.direction][action]
        new_coord = self._get_new_coord(self.snake[0], self.direction)
        self.snake.appendleft(new_coord)

        collision_result = self._is_collided(self.snake[0])
        done = False

        if collision_result == 1:
            self.score += 1
            self._generate_food()
        elif collision_result == 2:
            done = True
        else:
            self.snake.pop()

        state = self._get_game_state()
        reward = self._get_reward(collision_result)
        return state, reward, done

    def reset(self):
        self.score = 0
        self.snake.clear()
        start_x = (self.window_width // 2 // self.square_size) * self.square_size
        start_y = (self.window_height // 2 // self.square_size) * self.square_size
        self.snake.append((start_x, start_y))
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        self._generate_food()
        return self._get_game_state()

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        
        self.screen.fill(LIGHTER_GREEN)

        # Drawing checkered grid
        for x in range(self.margin, self.window_width - self.margin, self.square_size):
            for y in range(self.margin, self.window_height - self.margin, self.square_size):
                if ((x // self.square_size) + (y // self.square_size)) % 2 == 0:
                    color = DARKER_GREEN
                else:
                    color = LIGHTER_GREEN
                pygame.draw.rect(self.screen, color, 
                                 (x, y, self.square_size, self.square_size))

        pygame.draw.rect(self.screen, WHITE, 
                         (self.margin, self.margin, 
                          self.window_width - 2 * self.margin, 
                          self.window_height - 2 * self.margin), 1)

        font = pygame.font.SysFont("Courier New", 18, bold=True)
        score_text = font.render(f"SCORE: {self.score}", True, WHITE)
        self.screen.blit(score_text, [self.window_width // 2 - score_text.get_width() // 2, 0])

        for index, square_pos in enumerate(self.snake):
            color_to_use = SNAKE_DARKER if index == 0 else SNAKE_COLOR
            pygame.draw.rect(self.screen, color_to_use, 
                             (square_pos[0], square_pos[1], self.square_size, self.square_size))

        # Drawing apple
        if self.apple_image:
            self.screen.blit(self.apple_image, (self.food[0], self.food[1]))
        else:
            pygame.draw.rect(self.screen, RED,
                             (self.food[0], self.food[1], self.square_size, self.square_size))

        pygame.display.flip()

    def _generate_food(self):
        in_snake_body = True
        while in_snake_body:
            rand_x = random.randrange(self.margin, self.window_width - self.margin - self.square_size + 1, self.square_size)
            rand_y = random.randrange(self.margin, self.window_height - self.margin - self.square_size + 1, self.square_size)
            in_snake_body = any(rand_x == x and rand_y == y for x, y in self.snake)
        self.food = (rand_x, rand_y)

    def _is_collided(self, coord):
        x, y = coord
        if (x, y) == self.food:
            return 1
        
        if not (self.margin <= x < self.window_width - self.margin and
                self.margin <= y < self.window_height - self.margin):
            return 2

        if any(x == square_pos[0] and y == square_pos[1] for square_pos in list(self.snake)[1:]):
             return 2
        return 0

    def _get_reward(self, collision_result):
        rewards = {
            0: -0.1,
            1: 50.0,
            2: -100.0,
        }
        return rewards[collision_result]

    def _get_new_coord(self, coord, direction_str):
        x, y = coord
        if direction_str == "UP":    return (x, y - self.square_size)
        if direction_str == "DOWN":  return (x, y + self.square_size)
        if direction_str == "LEFT":  return (x - self.square_size, y)
        if direction_str == "RIGHT": return (x + self.square_size, y)
        return coord

    def _transform_coord(self, coord_diff, head_direction_str):
        x_diff, y_diff = coord_diff
        if head_direction_str == "UP":    return (x_diff, y_diff)
        if head_direction_str == "LEFT":  return (-y_diff, x_diff)
        if head_direction_str == "RIGHT": return (y_diff, -x_diff)
        if head_direction_str == "DOWN":  return (-x_diff, -y_diff)
        return coord_diff

    def _get_quadrant(self, coord_diff):
        def sign(n):
            if n < 0: return 1
            if n > 0: return 2
            return 0
        x_diff, y_diff = coord_diff
        return (sign(x_diff), sign(y_diff))

    def _get_game_state(self):
        state = []
        head_pos = self.snake[0]

        relative_directions = action_to_direction[self.direction]
        
        dir_straight = relative_directions[0]
        next_coord_straight = self._get_new_coord(head_pos, dir_straight)
        state.append(self._is_collided(next_coord_straight))

        dir_left = relative_directions[1]
        next_coord_left = self._get_new_coord(head_pos, dir_left)
        state.append(self._is_collided(next_coord_left))

        dir_right = relative_directions[2]
        next_coord_right = self._get_new_coord(head_pos, dir_right)
        state.append(self._is_collided(next_coord_right))
        
        food_dx = self.food[0] - head_pos[0]
        food_dy = self.food[1] - head_pos[1]
        
        transformed_food_coord = self._transform_coord((food_dx, food_dy), self.direction)
        food_quadrant_x, food_quadrant_y = self._get_quadrant(transformed_food_coord)
        state.append(food_quadrant_x)
        state.append(food_quadrant_y)
        
        return tuple(state)

    def final_show(self, agent, menu_options_str=""):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Snake RL - Trained Agent")

        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Courier New", 18, bold=True)

        while True:  # Outer loop to handle replay
            current_state = self.reset()
            done = False

            print("\n--- Playing Game with Trained Agent ---")
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Game aborted by user.")
                        pygame.quit()
                        sys.exit()

                action = agent.select_action(current_state, training=False)

                next_state, reward, done = self.step(action)
                current_state = next_state
                
                self.render()
                pygame.time.delay(max(1, 1000 // self.speed))

            # Game over
            font = pygame.font.SysFont("Arial Black", 24)
            game_over_text = font.render(f"Game Over! Score: {self.score}", True, WHITE)
            text_rect = game_over_text.get_rect(center=(self.window_width // 2, self.window_height // 3))
            pygame.draw.rect(self.screen, DARK_GRAY, text_rect.inflate(50, 25), border_radius=5)  
            self.screen.blit(game_over_text, text_rect)

            button_width, button_height = 100, 40
            play_button_rect = pygame.Rect(self.window_width // 2 - button_width - 20, self.window_height // 3 + 50, button_width, button_height)
            exit_button_rect = pygame.Rect(self.window_width // 2 + 20, self.window_height // 3 + 50, button_width, button_height)

            # Buttons
            font_buttons = pygame.font.SysFont("Verdana", 15)
            pygame.draw.rect(self.screen, GREEN_BUTTON, play_button_rect, border_radius=15)
            pygame.draw.rect(self.screen, BLACK, play_button_rect, 1, border_radius=15)
            pygame.draw.rect(self.screen, RED_BUTTON, exit_button_rect, border_radius=15)
            pygame.draw.rect(self.screen, BLACK, exit_button_rect, 1, border_radius=15)
            play_text = font_buttons.render("Play", True, BLACK)
            exit_text = font_buttons.render("Exit", True, BLACK)
            self.screen.blit(play_text, [play_button_rect.x + (button_width - play_text.get_width()) // 2, 
                                        play_button_rect.y + (button_height - play_text.get_height()) // 2])
            self.screen.blit(exit_text, [exit_button_rect.x + (button_width - exit_text.get_width()) // 2, 
                                        exit_button_rect.y + (button_height - exit_text.get_height()) // 2])

            pygame.display.flip()

            # Wait for user input
            waiting_for_choice = True
            while waiting_for_choice:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Game aborted by user.")
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = event.pos
                        if play_button_rect.collidepoint(mouse_pos):
                            waiting_for_choice = False  # Restart the game
                        elif exit_button_rect.collidepoint(mouse_pos):
                            print("Exiting game.")
                            pygame.quit()
                            sys.exit()
                clock.tick(10)