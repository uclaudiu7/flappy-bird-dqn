from ple.games.flappybird import FlappyBird
from ple import PLE
from agent import Agent

import torch
import copy
import cv2


def flappy_bird(agent):
    game = FlappyBird()
    game.allowed_fps = None
    p = PLE(game, fps=30, display_screen=True)
    p.init()
    actions = p.getActionSet()
    action_dict = {0: actions[1], 1: actions[0]}
    run = 0
    while True:
        run += 1
        p.reset_game()
        state = img_preprocess(p.getScreenRGB())
        step = 0

        while True:
            step += 1
            action, action_one_hot, p_values = agent.predict(state)
            reward = p.act(action_dict[action[0]])
            state_next = img_preprocess(p.getScreenRGB())
            terminal = p.game_over()

            reward = torch.tensor([reward])
            if torch.cuda.is_available():
                reward = reward.cuda()
                action_one_hot = action_one_hot.cuda()
            agent.remember(state, action_one_hot, reward, state_next, terminal)
            state = state_next

            if terminal:
                print("Episode: " + str(run) + ", score: " + str(step))
                break

            agent.experience_replay()

            if step % 50 == 0:
                agent.Q_target.load_state_dict(copy.deepcopy(agent.Q_policy.state_dict()))


def img_preprocess(img):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[:400, :]
    img = cv2.resize(img, (80, 80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
    return img


if __name__ == "__main__":
    action_space = 2
    agent = Agent(action_space)
    flappy_bird(agent)
