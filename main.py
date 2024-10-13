import gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
from qlearning_agent import QLearningAgent
from helpers import discretize

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 400
EPISODES = 1000
N_STATES = 20 * 20
N_ACTIONS = 3
TRIALS = 10

def run_experiment(alpha):
    env = gym.make('MountainCar-v0', render_mode=None)
    agent = QLearningAgent(env, n_states=N_STATES, n_actions=N_ACTIONS, alpha=alpha, gamma=0.99, epsilon=0.1)
    rewards = agent.train(EPISODES)
    env.close()
    return rewards, agent

def evaluate_agent(agent):
    env = gym.make('MountainCar-v0', render_mode=None)
    successes = []
    timesteps_per_trial = []
    for _ in range(TRIALS):
        state, _ = env.reset()
        done = False
        t = 0
        while not done:
            action = agent.choose_action(discretize(state))
            state, _, done, _, _ = env.step(action)
            t += 1
        timesteps_per_trial.append(t)
        successes.append(1 if t < 200 else 0)
    env.close()
    return successes, timesteps_per_trial

def draw_car(screen, x, y, color):
    pygame.draw.circle(screen, color, (int(x), int(y)), 5)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Mountain Car Q-Learning Comparison")

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {}
    agents = {}

    for alpha in alphas:
        print(f"Running experiment with alpha = {alpha}")
        rewards, agent = run_experiment(alpha)
        results[alpha] = rewards
        agents[alpha] = agent

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        for i, (alpha, agent) in enumerate(agents.items()):
            env = gym.make('MountainCar-v0', render_mode=None)
            state, _ = env.reset()
            x = (state[0] + 1.2) / 1.8 * (SCREEN_WIDTH / len(alphas) - 20) + i * (SCREEN_WIDTH / len(alphas)) + 10
            y = (1 - (state[1] + 0.07) / 0.14) * (SCREEN_HEIGHT - 20) + 10
            draw_car(screen, x, y, (255, 0, 0))
            action = agent.choose_action(discretize(state))
            state, _, done, _, _ = env.step(action)
            env.close()

            font = pygame.font.Font(None, 24)
            text = font.render(f"α = {alpha}", True, (0, 0, 0))
            screen.blit(text, (i * (SCREEN_WIDTH / len(alphas)) + 10, 10))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

    # Plot results
    plt.figure(figsize=(12, 6))
    for alpha, rewards in results.items():
        plt.plot(rewards, label=f'α = {alpha}')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning Performance for Different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate and plot statistics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for alpha, agent in agents.items():
        successes, timesteps = evaluate_agent(agent)
        
        ax1.bar(alpha, sum(successes)/len(successes), width=0.1)
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate for Different Alphas')

        ax2.bar(alpha, np.mean(timesteps), width=0.1)
        ax2.set_xlabel('Alpha')
        ax2.set_ylabel('Average Timesteps')
        ax2.set_title('Average Timesteps for Different Alphas')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
