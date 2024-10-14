# RLCode2-2


## Exemple complet

```
git clone https://github.com/hrhouma/RLCode2-2.git
cd RLCode2-2
python3 -m venv mountain_car (ici python3 et non python)
mountain_car\Scripts\activate
python nom_du_script.py  (Ici python et non python3)
pip install -r requirements.txt
python main.py
deactivate
```




# 1. `qlearning_agent.py`:

```python
import numpy as np
from helpers import discretize

class QLearningAgent:
    def __init__(self, env, n_states, n_actions, alpha, gamma, epsilon):
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def train(self, n_episodes):
        rewards_per_episode = []
        for episode in range(n_episodes):
            state = discretize(self.env.reset()[0])
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = discretize(next_state)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            rewards_per_episode.append(total_reward)
        return rewards_per_episode
```

# 2. `helpers.py`:

```python
import numpy as np
import pandas as pd

N_BINS = 20

position_bins = pd.cut([-1.2, 0.6], bins=N_BINS, retbins=True)[1][1:-1]
velocity_bins = pd.cut([-0.07, 0.07], bins=N_BINS, retbins=True)[1][1:-1]

def build_state(features):
    state_no = 0
    for i, feat in enumerate(features):
        state_no += (N_BINS ** i) * (feat - 1)
    return int(state_no)

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

def discretize(state):
    position, velocity = state
    return build_state([to_bin(position, position_bins),
                        to_bin(velocity, velocity_bins)])
```

# 3. `main.py`:

```python
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
```

# 4. `requirements.txt`:

```
numpy==1.23.5
pygame
pandas
gym
matplotlib
```

Pour exécuter cette solution :

1. Assurez-vous d'avoir tous ces fichiers dans le même dossier.
2. Installez les dépendances en exécutant `pip install -r requirements.txt`.
3. Exécutez `python main.py`.

Cette solution va :

1. Entraîner des agents Q-Learning avec différentes valeurs d'alpha.
2. Afficher une interface Pygame montrant les agents côte à côte dans l'environnement Mountain Car.
3. Après la fermeture de l'interface Pygame, afficher un graphique comparant les performances des différents agents pendant l'entraînement.
4. Enfin, afficher des statistiques sur le taux de succès et le nombre moyen de pas pour chaque valeur d'alpha.

Cette structure offre une démonstration visuelle et statistique de l'impact du taux d'apprentissage (alpha) sur les performances de l'agent Q-Learning dans l'environnement Mountain Car.

--------------

![image](https://github.com/user-attachments/assets/a80bab8e-1a12-422b-bc39-af6205c37d83)

--------------

![image](https://github.com/user-attachments/assets/30436b33-6064-4e80-9a35-8bda61537618)

--------------

![image](https://github.com/user-attachments/assets/0b0a49f6-a8e2-4eda-b406-5b883190555a)

--------------

![image](https://github.com/user-attachments/assets/6f168b68-7329-4400-8f00-24796e2e197f)




---------------------------
# Annexe :  🚗 **Projet MountainCar avec Q-Learning et Comparaison des Alphas** 🚗
---------------------------

## 🛠️ **Introduction**

Ce projet utilise l'algorithme **Q-Learning** pour résoudre le problème du **MountainCar-v0** avec plusieurs valeurs de **taux d'apprentissage alpha (α)**. Il permet d'observer comment différentes valeurs d'**alpha** affectent les performances de l'agent lors de l'apprentissage. Le programme offre à la fois un rendu visuel avec **`pygame`** et des graphiques pour analyser les résultats.

### 🔑 **Alpha (α) : Taux d'apprentissage**
- **α** est un paramètre crucial dans l'algorithme Q-Learning. Il détermine l'importance accordée aux nouvelles expériences par rapport aux connaissances précédentes.
- Le projet teste les valeurs suivantes : **0.1, 0.3, 0.5, 0.7, 0.9**.

---

## 🧠 **Explication du processus** :

1. **Entraînement de l'agent avec différentes valeurs de α :**
   - Chaque agent est entraîné à résoudre **MountainCar-v0** avec une valeur spécifique d'**alpha** pendant **1000 épisodes**. 
   - L'objectif est d'atteindre le sommet de la colline en optimisant ses actions.

2. **Évaluation des agents :**
   - Une fois entraînés, les agents sont évalués sur **10 essais**. On mesure le nombre de pas nécessaires pour réussir et le taux de réussite (atteindre le sommet en moins de 200 pas).
   
3. **Visualisation en temps réel avec Pygame :**
   - Un rendu **pygame** est généré où chaque agent est représenté par une voiture dans l'environnement. Chaque voiture agit selon l'agent entraîné avec une valeur spécifique d'**alpha**.
   
4. **Affichage des résultats :**
   - À la fin de la simulation, deux graphiques sont générés :
     1. **Récompenses cumulées pendant l'entraînement** pour chaque valeur d'**alpha**.
     2. **Taux de réussite et moyenne des pas de temps** après l'évaluation pour chaque agent.

---

## 🖼️ **Ce que vous allez voir à la fin de l'exécution du code**

### **1. Rendu visuel avec `pygame`**

Dans la première partie, vous verrez un **rendu visuel dynamique** de plusieurs agents s'entraînant en parallèle avec différentes valeurs d'**alpha**.

- Chaque voiture (point rouge) représente un agent, et chaque section de l'écran est dédiée à un **alpha** spécifique, comme **α = 0.1**, **α = 0.3**, **α = 0.5**, etc.
- Vous pourrez observer comment chaque voiture se comporte en fonction de la valeur d'**alpha** pendant qu'elle tente de résoudre l'environnement MountainCar.

### **2. Graphiques des résultats après l'évaluation**

À la fin de l'exécution, deux graphiques seront générés :

1. **Graphique du taux de succès** : Il montre le pourcentage d'essais réussis par chaque agent selon la valeur de son **alpha**.
   - Exemple : **α = 0.3** semble obtenir le meilleur taux de succès dans ce cas.
   
2. **Graphique du nombre moyen de pas de temps** : Ce graphique montre combien de pas en moyenne l'agent prend pour réussir.
   - Exemple : **α = 0.9** nécessite beaucoup plus de pas, ce qui signifie que l'agent n'a pas bien appris avec cette valeur.

---

## 📊 **Tableau de comparaison des performances des alphas (α)**

Voici un tableau ASCII pour récapituler la **comparaison des performances** des différentes valeurs d'alpha en termes de taux de succès et du nombre moyen de pas nécessaires :

```
+---------+----------------------+---------------------------+
| Alpha   | Taux de réussite (%) | Nombre moyen de pas de temps|
+---------+----------------------+---------------------------+
| α = 0.1 |        50%           |        ~1,500              |
| α = 0.3 |        85%           |         150                |
| α = 0.5 |        70%           |         300                |
| α = 0.7 |        60%           |        ~500                |
| α = 0.9 |        10%           |       > 10,000             |
+---------+----------------------+---------------------------+
```

### 📝 **Analyse :**
- **α = 0.3** donne les meilleurs résultats avec un taux de réussite élevé et un faible nombre moyen de pas de temps.
- **α = 0.9** montre que l'agent explore trop et n'arrive pas à stabiliser son apprentissage, nécessitant beaucoup trop de pas.
- Les **alphas faibles**, comme **α = 0.1**, sont plus lents à apprendre mais montrent une certaine stabilité.

---

## 🏁 **Conclusion :**
- **Rendu visuel en temps réel** : Vous verrez les agents en action pendant qu'ils essaient de résoudre le problème MountainCar, chaque agent ayant une stratégie différente en fonction de la valeur d'**alpha**.
- **Graphiques de performance** : À la fin de l'exécution, des graphiques vous montreront les performances de chaque agent, en termes de **taux de réussite** et de **nombre de pas de temps** nécessaires pour réussir.

Ces résultats vous permettront d'analyser comment différentes valeurs d'**alpha** influencent l'apprentissage et la performance des agents Q-Learning, vous offrant une **vue pédagogique claire** sur le rôle du **taux d'apprentissage** dans un environnement dynamique.

