# Flappy Bird Deep Q-Network (DQN) Agent

## Overview

This project implements a Deep Q-Network (DQN) agent to play the Flappy Bird game using reinforcement learning. The agent learns to navigate through the game environment by training on a neural network that approximates the Q-function.

## Dependencies

- Python 3.x
- Pygame Learning Environment (PLE)
- PyTorch
- OpenCV

## Usage
1. Install PLE (PyGame Learning Environment)
   * clone the repo:
     ```bash
     git clone https://github.com/ntasfi/PyGame-Learning-Environment
   * use the cd command to enter the PLE directory and run the command:
     ```bash
     sudo pip install -e .
3. Install other dependencies:
    ```bash
    pip install torch opencv-python

4. Run the flappy_bird.py script:
    ```bash
    python flappy_bird.py

## Project Structure
* flappy_bird.py: Main script to run the Flappy Bird game with the DQN agent.
* agent.py: Defines the Agent class responsible for training and experience replay.
* dqn.py: Defines the neural network architecture (DQN) used as the Q-function approximator.

## Customization
Feel free to customize the code to experiment with different hyperparameters, network architectures, or training strategies. You can also adapt the code to work with other environments.

## Credits
Flappy Bird game environment: [Pygame Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html)
