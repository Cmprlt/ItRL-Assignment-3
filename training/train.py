import os
import random
import csv
from collections import deque
import torch
import torch.optim as optim
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config.settings import Config
from models.q_network import QNetwork


def to_tensor(state):
    '''
    Helper function to make an environment state (4 numbers) into a tensor of
    [1,4] shape
    '''
    return torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)


def epsilon_at(step, config):
    '''
    Function returns the exploration rate at the given time step.
    Starts at eps_start where it is fully random and then decays down to eps_end
    '''
    fraction = min(1.0, step / config.eps_decay_steps)
    return config.eps_start + fraction * (config.eps_end - config.eps_start)


def choose_action(q_network, state, epsilon):
    '''
    Function returns random action (0 or 1) if probability smaller than epsilon, otherwise
    highest Q-value action is chosen.
    torch.no_grad() is used because choosing an action doesn't update the network
    so the gradients are not needed
    '''
    if random.random() < epsilon:
        return random.randrange(2)
    with torch.no_grad():
        return q_network(state).argmax().item()


def learn(q_network, target_network, optimizer, buffer, config):
    '''
    Function for one update for q_network from a random batch of remembered steps.
    Each buffer item is = state, action, reward, next_state, terminated
    where each state and next_state are tensors of shape [1,4]
    '''

    if len(buffer) < config.batch_size:         # have to have enough memories stored
        return

    # random batch, each part stacked into one tensor
    batch = random.sample(buffer, config.batch_size)
    states = torch.cat([b[0] for b in batch])                           # [64, 4]
    actions = torch.tensor([b[1] for b in batch]).unsqueeze(1)          # [64, 1]
    rewards = torch.tensor([b[2] for b in batch]).unsqueeze(1)          # [64, 1]
    next_states = torch.cat([b[3] for b in batch])                      # [64, 4]
    terminated = torch.tensor([float(b[4]) for b in batch]).unsqueeze(1)# [64, 1]

    # Q-network's Q-value for the action that was actually taken
    guesses = q_network(states).gather(1, actions)

    # target = reward + gamma * best Q-value of the next state, from the slow
    # target network. (1 - terminated) removes the future part if the pole fell.
    with torch.no_grad():
        best_next = target_network(next_states).max(dim=1, keepdim=True)[0]
        targets = rewards + config.gamma * best_next * (1 - terminated)

    # move the guesses towards the targets.
    loss = ((guesses - targets) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # let the target network follow q_network by a small step
    with torch.no_grad():
        for t, q in zip(target_network.parameters(), q_network.parameters()):
            t.mul_(1 - config.tau).add_(config.tau * q)


def save_training_logs_csv(name, training_logs):
    os.makedirs("data", exist_ok=True)
    log_path = os.path.join("data", f"{name}_training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["episode", "total_episodes", "average_of_last_20", "epsilon"])
        for entry in training_logs:
            writer.writerow([
                entry["episode"],
                entry["total_episodes"],
                f"{entry['average_20']:.1f}",
                f"{entry['epsilon']:.2f}",
            ])
    return log_path


def plot_training_logs(name, training_logs):
    os.makedirs("data", exist_ok=True)
    plot_path = os.path.join("data", f"{name}_training_plot.png")
    episodes = []
    averages = []
    epsilons = []
    for entry in training_logs:
        episodes.append(entry["episode"])
        averages.append(entry["average_20"])
        epsilons.append(entry["epsilon"])

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if episodes:
        axes[0].plot(episodes, averages, marker="o", label="Average of last 20 episodes")
        axes[1].plot(episodes, epsilons, color="tab:orange", marker="o", label="Epsilon")
    else:
        axes[0].text(0.5, 0.5, "No 25-episode summaries were recorded", ha="center", va="center")
        axes[0].set_xlim(0, 1)
    axes[0].set_ylabel("Average score")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Epsilon")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    figure.suptitle(f"{name} training logs")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path




def train(name, config=None, sampler=None, episode_callback=None):
    '''
    Function trains the standart QNetwork provided with DQN and saves the weights
    as weights/<name>.pth.
    Args:
    name = name of the file to save like "baseline_seed0" for example
    config = specify lenght range for the poles and seed, defaults to config.py if not set
    sampler is an optional argument = picks the pole length for every episode and (check samplers.py)
    if it is not set lengths come from config.py
    episode_callback = optional callable invoked after each episode (and after
    sampler.update, when a sampler is used). Receives a dict with episode,
    pole_length, steps_survived, and epsilon. Defaults to None so existing
    strategies are unchanged.
    '''
    config = config or Config()

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    env = gym.make("CartPole-v1")

    q_network = QNetwork(4, 2)                      # what is trained and saved
    target_network = QNetwork(4, 2)                 # the copy for targets only
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=config.learning_rate)
    buffer = deque(maxlen=config.buffer_size)   # drops the oldest when full

    os.makedirs("weights", exist_ok=True)
    path = os.path.join("weights", f"{name}.pth")
    total_steps, history, best_average = 0, [], 0.0
    training_logs = []

    for episode in range(config.episodes):
        #sampler picks the pole lenght for the episode if set
        if sampler:
            length = sampler.sample()
        else:
            length = random.uniform(*config.length_range)
        env.unwrapped.length = length

        state, _ = env.reset(seed=config.seed if episode == 0 else None)
        state = to_tensor(state)
        steps = 0
        terminated = False
        truncated = False

        # play one episode, learning after every step
        while not (terminated or truncated):
            epsilon = epsilon_at(total_steps, config)
            action = choose_action(q_network, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = to_tensor(next_state)
            buffer.append((state, action, float(reward), next_state, terminated)) # store only terminated because only a fall doesn't have a future
            learn(q_network, target_network, optimizer, buffer, config)

            state = next_state
            steps += 1
            total_steps += 1

        history.append(steps)
        average = sum(history[-20:]) / len(history[-20:])
        training_logs.append({"episode": episode + 1, "length": length, "steps": steps, "total_episodes": config.episodes, "average_20": average, "epsilon": epsilon})

        if sampler: #back to sampler - how many steps survived at this sampled length 
            sampler.update(length, steps)

        if episode_callback is not None:
            episode_callback({
                "episode": episode + 1,
                "pole_length": length,
                "steps_survived": steps,
                "epsilon": epsilon,
            })

        # keep the best checkpoint only as the DQN can get worse later in training
        if len(history) >= 20 and average > best_average:
            best_average = average
            torch.save(q_network.state_dict(), path)

        if (episode + 1) % 25 == 0:
            print(f"[{name}] episode {episode + 1}/{config.episodes}  "
                  f"average of last 20: {average:.1f}  epsilon {epsilon:.2f}")

    env.close()
    save_training_logs_csv(name, training_logs)
    plot_training_logs(name, training_logs)
    if best_average > 0:
        print(f"[{name}] saved {path} (best average {best_average:.1f})")
    else:
        print(f"[{name}] nothing saved, less than 20 episodes ran")

    return path

