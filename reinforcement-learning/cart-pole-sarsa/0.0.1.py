# /// script
# title = "CartPole SARSA Optimization"
# description = "Visualize the CartPole game optimized using SARSA in Napari."
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.0.1"
# keywords = ["cartpole", "sarsa", "reinforcement learning", "napari", "interactive"]
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Developers",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.11"
# dependencies = [
#     "napari[all]",
#     "numpy>=1.24.0",
#     "typer",
#     "gym",
#     "opencv-python",
#     "magicgui",
# ]
# ///

# Key fixes:
# 1. Added proper yield timing control
# 2. Enhanced frame update mechanism
# 3. Improved worker management
# 4. Added error handling for frame updates

import napari
import numpy as np
np.bool8 = bool
import gym
import cv2
from magicgui import magicgui
from superqt.utils import thread_worker
import time
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SimulationState:
    def __init__(self):
        self.paused = False
        self.worker = None
        self.frame_delay = 0.05  # Added control for frame timing
        logger.debug("SimulationState initialized")

    def toggle_pause(self):
        self.paused = not self.paused
        logger.debug(f"Simulation paused state: {self.paused}")

    def is_paused(self):
        return self.paused

    def stop_worker(self):
        if self.worker is not None:
            logger.debug("Stopping worker")
            try:
                self.worker.quit()
                self.worker.wait()  # Wait for worker to finish
            except Exception as e:
                logger.error(f"Error stopping worker: {e}")
            self.worker = None

    def set_worker(self, worker):
        self.stop_worker()
        self.worker = worker
        self.worker.start()
        logger.debug("New worker started")

class SARSA_Agent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_table = np.zeros(n_states + (n_actions,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.episode_rewards = []
        self.episode_lengths = []
        logger.debug(f"SARSA Agent initialized with shape {self.q_table.shape}")

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            logger.debug(f"Action chosen: {action} (random)")
        else:
            action = np.argmax(self.q_table[state])
            logger.debug(f"Action chosen: {action} (greedy), Q-values: {self.q_table[state]}")
        return action

    def update(self, state, action, reward, next_state, next_action):
        old_value = self.q_table[state][action]
        td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_error = td_target - old_value
        self.q_table[state][action] += self.alpha * td_error
        logger.debug(f"Q-value updated: {old_value:.3f} -> {self.q_table[state][action]:.3f}")

@thread_worker
def frame_generator(env, agent, state, action, n_bins, max_steps, state_manager):
    step_count = 0
    episode_reward = 0
    last_frame_time = time.time()
    episode_count = 0
    max_episodes = 100  # Run for multiple episodes
    
    try:
        while episode_count < max_episodes:
            current_time = time.time()
            if not state_manager.is_paused() and (current_time - last_frame_time) >= state_manager.frame_delay:
                step_result = env.step(action)
                
                if isinstance(step_result, tuple):
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state = step_result["observation"]
                    reward = step_result["reward"]
                    done = step_result["terminated"] or step_result["truncated"]
                
                episode_reward += reward
                next_state_discrete = discretize_state(next_state, n_bins)
                next_action = agent.choose_action(next_state_discrete)
                
                agent.update(state, action, reward, next_state_discrete, next_action)
                
                frame = render_cartpole(next_state)
                yield frame
                
                if done:
                    logger.debug(f"Episode {episode_count} finished after {step_count} steps, reward: {episode_reward}")
                    agent.episode_rewards.append(episode_reward)
                    agent.episode_lengths.append(step_count)
                    agent.decay_epsilon()
                    
                    # Reset for next episode
                    episode_count += 1
                    state_dict = env.reset()
                    if isinstance(state_dict, tuple):
                        next_state = state_dict[0]
                    else:
                        next_state = state_dict["observation"]
                    next_state_discrete = discretize_state(next_state, n_bins)
                    next_action = agent.choose_action(next_state_discrete)
                    episode_reward = 0
                    step_count = 0
                else:
                    state, action = next_state_discrete, next_action
                    step_count += 1
                
                if step_count % 100 == 0:  # Log every 100 steps
                    logger.debug(f"State distribution: {state}")
                    logger.debug(f"Cart position bin: {state[0]}")


                last_frame_time = current_time
            
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error in frame generator: {e}")
        yield np.zeros((100, 200), dtype=np.uint8)

def discretize_state(state, n_bins):
    bins = [
        np.linspace(-4.8, 4.8, n_bins),
        np.linspace(-4, 4, n_bins),
        np.linspace(-0.418, 0.418, n_bins),
        np.linspace(-4, 4, n_bins),
    ]
    discrete_state = tuple(np.digitize(state[i], bins[i]) - 1 for i in range(len(state)))
    return discrete_state

def render_cartpole(state, canvas_shape=(100, 200)):
    try:
        # Center the cart's range in the middle of the canvas
        canvas_center = canvas_shape[1] // 2
        pixels_per_unit = (canvas_shape[1] - 40) / 4.8  # Leave 20px margin on each side
        cart_position = int(canvas_center + state[0] * pixels_per_unit)
        
        pole_angle = state[2]
        canvas = np.zeros(canvas_shape, dtype=np.uint8)
        
        # Draw the cart centered on its position
        cart_width = 20
        cv2.rectangle(canvas, 
                     (cart_position - cart_width//2, 80),
                     (cart_position + cart_width//2, 90),
                     255, -1)
        
        # Draw the pole from the cart's center
        pole_length = 50
        pole_x = int(cart_position + pole_length * np.sin(pole_angle))
        pole_y = int(85 - pole_length * np.cos(pole_angle))
        cv2.line(canvas, (cart_position, 85), (pole_x, pole_y), 128, 2)
        
        return canvas
    except Exception as e:
        logger.error(f"Error in render_cartpole: {e}")
        return np.zeros(canvas_shape, dtype=np.uint8)

def start_worker(env, agent, state, action, n_bins, max_steps, state_manager, update_layer):
    logger.debug("Starting new worker")
    worker = frame_generator(env, agent, state, action, n_bins, max_steps, state_manager)
    
    def on_yield(frame):
        try:
            update_layer(frame)
        except Exception as e:
            logger.error(f"Error in frame update: {e}")
    
    worker.yielded.connect(on_yield)
    state_manager.set_worker(worker)

def run():
    viewer = napari.Viewer()
    canvas_shape = (100, 200)
    canvas_layer = viewer.add_image(np.zeros(canvas_shape), name="CartPole")
    
    env = gym.make("CartPole-v1")
    n_bins = 10
    max_steps = 200
    alpha, gamma, epsilon = 0.1, 0.99, 0.1
    
    agent = SARSA_Agent((n_bins,) * len(env.observation_space.high), env.action_space.n, alpha, gamma, epsilon)
    
    state_dict = env.reset()
    if isinstance(state_dict, tuple):
        state = state_dict[0]
    else:
        state = state_dict["observation"]
    
    state = discretize_state(state, n_bins)
    action = agent.choose_action(state)
    
    state_manager = SimulationState()

    @magicgui
    def update_canvas(frame):
        try:
            canvas_layer.data = frame
        except Exception as e:
            logger.error(f"Error updating canvas: {e}")

    class ControlWidget(QWidget):
        def __init__(self, state_manager):
            super().__init__()
            self.state_manager = state_manager
            self.initUI()

        def initUI(self):
            layout = QVBoxLayout()
            
            self.status_label = QLabel("Status: Running")
            layout.addWidget(self.status_label)

            self.pause_button = QPushButton("Pause")
            self.pause_button.clicked.connect(self.toggle_pause)
            layout.addWidget(self.pause_button)

            self.reset_button = QPushButton("Reset")
            self.reset_button.clicked.connect(self.reset_simulation)
            layout.addWidget(self.reset_button)

            self.setLayout(layout)

        def toggle_pause(self):
            self.state_manager.toggle_pause()
            status = "Paused" if self.state_manager.is_paused() else "Running"
            self.status_label.setText(f"Status: {status}")
            self.pause_button.setText("Resume" if self.state_manager.is_paused() else "Pause")

        def reset_simulation(self):
            nonlocal state, action
            state_dict = env.reset()
            if isinstance(state_dict, tuple):
                state = discretize_state(state_dict[0], n_bins)
            else:
                state = discretize_state(state_dict["observation"], n_bins)
            action = agent.choose_action(state)
            start_worker(env, agent, state, action, n_bins, max_steps, self.state_manager, update_canvas)

    control_widget = ControlWidget(state_manager)
    viewer.window.add_dock_widget(control_widget, area="right")
    
    start_worker(env, agent, state, action, n_bins, max_steps, state_manager, update_canvas)
    
    napari.run()

if __name__ == "__main__":
    run()