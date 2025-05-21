import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpe2 import simple_spread_v3
from torch.distributions import Categorical
from collections import deque
import random

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network Models for Custom Algorithm
class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class GNNMessageModel(nn.Module):
    def __init__(self, latent_dim, message_dim, num_agents):
        super(GNNMessageModel, self).__init__()
        self.num_agents = num_agents
        self.encoder = nn.Linear(latent_dim, 32)
        self.message_layer = nn.Linear(32, message_dim)
        self.update_layer = nn.Linear(32 + message_dim, 32)
        self.out = nn.Linear(32, message_dim)
    
    def forward(self, z, tau, agent_dists):
        h = F.relu(self.encoder(z))
        
        # Build adjacency matrix based on agent distances
        adj = torch.zeros(self.num_agents, self.num_agents, device=device)
        threshold = 1.5  # Agents within 1.5 units communicate
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j and agent_dists[i, j] < threshold:
                    adj[i, j] = 1.0
        
        # Message passing
        messages = self.message_layer(h)
        aggregated_msgs = torch.matmul(adj, messages)
        h = F.relu(self.update_layer(torch.cat([h, aggregated_msgs], dim=-1)))
        messages = self.out(h)
        messages = F.gumbel_softmax(messages, tau=tau, hard=False, dim=-1)
        return messages

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, message_dim, latent_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim + message_dim + latent_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
    
    def forward(self, obs, messages, z):
        x = torch.cat([obs, messages, z], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        action_logits = self.fc3(h)
        return F.softmax(action_logits, dim=-1)

class CentralCritic(nn.Module):
    def __init__(self, global_obs_dim):
        super(CentralCritic, self).__init__()
        self.fc1 = nn.Linear(global_obs_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

# Environment Setup
num_agents = 3
num_episodes = 1500
max_steps = 75
obs_dim = 18
action_dim = 5

# Training Parameters
initial_lr = 0.0005
final_lr = 0.0001
lr_decay = (initial_lr - final_lr) / num_episodes
initial_epsilon = 0.3
final_epsilon = 0.1
epsilon_decay = (initial_epsilon - final_epsilon) / num_episodes

# Custom Algorithm
class CustomAlgorithm:
    def __init__(self):
        self.encoder = Encoder(obs_dim, 16).to(device)
        self.message_model = GNNMessageModel(16, 8, num_agents).to(device)
        self.policy = PolicyNetwork(obs_dim, 8 * (num_agents - 1), 16, action_dim).to(device)
        self.critic = CentralCritic(obs_dim * num_agents).to(device)
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.message_model.parameters()) + list(self.policy.parameters()), lr=initial_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=initial_lr)
        self.replay_buffer = deque(maxlen=10000)
        self.initial_tau = 2.0
        self.final_tau = 0.7
        self.tau_decay = (self.initial_tau - self.final_tau) / num_episodes
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

    def adjust_exploration(self, episode, landmarks_covered):
        base_epsilon = max(self.final_epsilon, self.initial_epsilon - episode * self.epsilon_decay)
        base_tau = max(self.final_tau, self.initial_tau - episode * self.tau_decay)
        coverage_factor = 1.0 - landmarks_covered  # Higher when coverage is low
        adjusted_epsilon = base_epsilon + 0.2 * coverage_factor
        adjusted_tau = base_tau + 0.5 * coverage_factor
        return adjusted_epsilon, adjusted_tau

    def act(self, obs, episode, landmarks_covered):
        adjusted_epsilon, current_tau = self.adjust_exploration(episode, landmarks_covered)
        # Convert obs to a single NumPy array, then to a PyTorch tensor
        obs_array = np.array(obs, dtype=np.float32)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(device)
        mu, logvar = self.encoder(obs_tensor)
        z = self.encoder.reparameterize(mu, logvar)
        
        # Compute agent distances for GNN using PyTorch operations
        agent_dists = torch.zeros(num_agents, num_agents, device=device)
        other_agent_start = 4 + 2 * 3
        for i in range(num_agents):
            for j in range(num_agents - 1):
                idx = j if j < i else j + 1
                x_rel = obs_tensor[i, other_agent_start + 2 * j]
                y_rel = obs_tensor[i, other_agent_start + 2 * j + 1]
                dist = torch.sqrt(x_rel**2 + y_rel**2)
                agent_dists[i, idx] = dist
                agent_dists[idx, i] = dist
        
        messages = self.message_model(z.unsqueeze(0), current_tau, agent_dists).squeeze(0)
        actions = []
        log_probs = []
        for i in range(num_agents):
            other_messages = torch.cat([messages[j] for j in range(num_agents) if j != i], dim=0)
            action_probs = self.policy(obs_tensor[i:i+1], other_messages.unsqueeze(0), z[i:i+1])
            action_dist = Categorical(probs=action_probs)
            if np.random.rand() < adjusted_epsilon:
                action = torch.randint(0, action_dim, (1,), device=device)
            else:
                action = action_dist.sample()
            actions.append(action.item())
            log_probs.append(action_dist.log_prob(action))
        return actions, log_probs

    def store_transition(self, obs, log_probs, reward):
        # Detach log_probs to prevent storing the computational graph
        detached_log_probs = [log_prob.detach() for log_prob in log_probs]
        self.replay_buffer.append((obs, detached_log_probs, reward))

    def update(self, log_probs, reward, obs):
        self.store_transition(obs, log_probs, reward)
        
        if len(self.replay_buffer) < 128:
            return  # Wait until buffer has enough samples
        
        # Sample a batch
        batch = random.sample(self.replay_buffer, 64)
        batch_obs, batch_log_probs, batch_rewards = zip(*batch)
        
        # Convert to tensors
        batch_obs_array = np.array(batch_obs, dtype=np.float32)
        batch_obs_tensor = torch.tensor(batch_obs_array, dtype=torch.float32).to(device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
        
        # Compute policy loss on batch (batch_log_probs are already detached)
        policy_loss = 0
        for i in range(len(batch)):
            log_prob = torch.stack(batch_log_probs[i])
            policy_loss += -log_prob.mean() * batch_rewards[i]
        policy_loss /= len(batch)
        
        # Compute policy loss for the current step (using current log_probs)
        current_policy_loss = -torch.stack(log_probs).mean() * torch.tensor(reward, dtype=torch.float32).to(device)
        
        # Combine policy losses (current step + replay buffer)
        total_policy_loss = (current_policy_loss + policy_loss) / 2
        
        # Critic loss
        global_obs = batch_obs_tensor.view(len(batch), -1)
        predicted_value = self.critic(global_obs)
        value_loss = F.mse_loss(predicted_value.squeeze(), batch_rewards)
        
        # KL divergence
        mu, logvar = self.encoder(batch_obs_tensor)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # Combined loss
        loss = total_policy_loss + 0.1 * value_loss + 0.05 * kl_div
        
        # Ensure gradients are cleared before backpropagation
        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.critic_optimizer.step()

# Training Loop
algo = CustomAlgorithm()

# Metrics storage
rewards_history = []
raw_rewards_history = []
shaped_rewards_history = []
collision_history = []
agent_distances_history = []
landmark_dist_history = []
landmarks_covered_history = []

# Curriculum for local_ratio and coverage threshold
initial_local_ratio = 1.0
final_local_ratio = 0.5
local_ratio_decay = (initial_local_ratio - final_local_ratio) / num_episodes
initial_coverage_threshold = 0.5
final_coverage_threshold = 0.15
threshold_decay = (initial_coverage_threshold - final_coverage_threshold) / num_episodes

for episode in range(num_episodes):
    current_local_ratio = max(final_local_ratio, initial_local_ratio - (episode * local_ratio_decay))
    current_coverage_threshold = max(final_coverage_threshold, initial_coverage_threshold - episode * threshold_decay)
    env = simple_spread_v3.parallel_env(N=3, max_cycles=max_steps, continuous_actions=False, local_ratio=current_local_ratio)
    obs_dict, infos = env.reset()

    # Update learning rate
    current_lr = max(final_lr, initial_lr - (episode * lr_decay))
    for optimizer in [algo.optimizer, algo.critic_optimizer]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    episode_reward = 0
    episode_collisions = 0
    episode_raw_rewards = []
    episode_shaped_rewards = []
    episode_agent_distances = []
    episode_landmark_dists = []
    episode_landmarks_covered = 0

    for step in range(max_steps):
        obs = [obs_dict[agent] for agent in env.agents]
        # Pass landmarks_covered for adaptive exploration
        landmarks_covered_last = episode_landmarks_covered / (step + 1) if step > 0 else 0.0
        actions, log_probs = algo.act(obs, episode, landmarks_covered_last)
        action_dict = {agent: actions[i] for i, agent in enumerate(env.agents)}

        # Compute distances between agents
        other_agent_start = 4 + 2 * 3
        agent_dists = []
        for agent_idx, agent in enumerate(env.agents):
            obs_agent = obs_dict[agent]
            for other_idx in range(num_agents - 1):
                x_rel = obs_agent[other_agent_start + 2 * other_idx]
                y_rel = obs_agent[other_agent_start + 2 * other_idx + 1]
                dist = np.sqrt(x_rel**2 + y_rel**2)
                agent_dists.append(dist)
        avg_agent_dist = np.mean(agent_dists) if agent_dists else 0.0
        episode_agent_distances.append(avg_agent_dist)

        # Compute average distance to closest landmark
        landmark_rel_pos_start = 4
        n_landmarks = 3
        min_landmark_dists = []
        landmarks_covered = 0
        for agent_idx, agent in enumerate(env.agents):
            obs_agent = obs_dict[agent]
            landmark_dists = []
            for l in range(n_landmarks):
                x_rel = obs_agent[landmark_rel_pos_start + 2*l]
                y_rel = obs_agent[landmark_rel_pos_start + 2*l + 1]
                dist = np.sqrt(x_rel**2 + y_rel**2)
                landmark_dists.append(dist)
            min_dist = min(landmark_dists)
            min_landmark_dists.append(min_dist)
            if min_dist < current_coverage_threshold:
                landmarks_covered += 1
        avg_min_landmark_dist = np.mean(min_landmark_dists)
        episode_landmark_dists.append(avg_min_landmark_dist)
        episode_landmarks_covered += landmarks_covered / (num_agents * n_landmarks)

        # Step environment
        next_obs_dict, rewards, dones, truncs, infos = env.step(action_dict)

        # Compute raw reward
        raw_reward = sum(rewards.values()) / num_agents
        episode_raw_rewards.append(raw_reward)

        # Estimate collisions
        collision_threshold = 0.2
        est_collisions = sum(1 for dist in agent_dists if dist < collision_threshold) / num_agents
        episode_collisions += est_collisions

        # Enhanced reward shaping with scaled coverage bonus
        shaped_reward = raw_reward
        for agent_idx, agent in enumerate(env.agents):
            obs_agent = obs_dict[agent]
            landmark_dists = []
            for l in range(n_landmarks):
                x_rel = obs_agent[landmark_rel_pos_start + 2*l]
                y_rel = obs_agent[landmark_rel_pos_start + 2*l + 1]
                dist = np.sqrt(x_rel**2 + y_rel**2)
                landmark_dists.append(dist)
            min_dist = min(landmark_dists)
            proximity_bonus = 10.0 * (1.0 - min_dist / 2.0)**2
            shaped_reward += proximity_bonus
            if min_dist < current_coverage_threshold:
                coverage_bonus = 3.0 * (1.0 - min_dist / current_coverage_threshold)
                shaped_reward += coverage_bonus

        # Penalty for collisions
        collision_penalty = 0.0
        if avg_agent_dist < collision_threshold:
            collision_penalty = -1.0
        shaped_reward += collision_penalty

        episode_reward += shaped_reward
        episode_shaped_rewards.append(shaped_reward)

        # Update the algorithm
        algo.update(log_probs, shaped_reward, obs)

        obs_dict = next_obs_dict
        if any(dones.values()) or any(truncs.values()):
            break

    # Store metrics
    rewards_history.append(episode_reward / max_steps)
    raw_rewards_history.append(np.mean(episode_raw_rewards))
    shaped_rewards_history.append(np.mean(episode_shaped_rewards))
    collision_history.append(episode_collisions / max_steps)
    agent_distances_history.append(np.mean(episode_agent_distances))
    landmark_dist_history.append(np.mean(episode_landmark_dists))
    landmarks_covered_history.append(episode_landmarks_covered / max_steps)

    # Print progress
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, "
              f"Avg Shaped Reward: {rewards_history[-1]:.2f}, "
              f"Collisions (est): {collision_history[-1]:.2f}, "
              f"Raw Reward: {raw_rewards_history[-1]:.2f}, "
              f"Shaped Reward: {shaped_rewards_history[-1]:.2f}, "
              f"Avg Agent Distance: {agent_distances_history[-1]:.2f}, "
              f"Avg Min Landmark Dist: {landmark_dist_history[-1]:.2f}, "
              f"Landmarks Covered: {landmarks_covered_history[-1]:.2f}")

# Visualization
plt.figure(figsize=(15, 15))

# Plot 1: Shaped Reward
plt.subplot(3, 2, 1)
smoothed = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
plt.plot(smoothed, label='Custom', color='blue')
plt.title("Shaped Reward Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Average Shaped Reward")
plt.grid(True)
plt.legend()

# Plot 2: Raw Reward
plt.subplot(3, 2, 2)
smoothed = np.convolve(raw_rewards_history, np.ones(10)/10, mode='valid')
plt.plot(smoothed, label='Custom', color='blue')
plt.title("Raw Reward Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Raw Reward")
plt.grid(True)
plt.legend()

# Plot 3: Landmark Coverage
plt.subplot(3, 2, 3)
smoothed = np.convolve(landmarks_covered_history, np.ones(10)/10, mode='valid')
plt.plot(smoothed, label='Custom', color='blue')
plt.title("Landmarks Covered Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Landmarks Covered")
plt.grid(True)
plt.legend()

# Plot 4: Min Landmark Distance
plt.subplot(3, 2, 4)
smoothed = np.convolve(landmark_dist_history, np.ones(10)/10, mode='valid')
plt.plot(smoothed, label='Custom', color='blue')
plt.title("Min Landmark Distance Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Avg Min Landmark Distance")
plt.grid(True)
plt.legend()

# Plot 5: Collisions
plt.subplot(3, 2, 5)
smoothed = np.convolve(collision_history, np.ones(10)/10, mode='valid')
plt.plot(smoothed, label='Custom', color='blue')
plt.title("Collisions Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Collisions (est)")
plt.grid(True)
plt.legend()

# Plot 6: Avg Agent Distance
plt.subplot(3, 2, 6)
smoothed = np.convolve(agent_distances_history, np.ones(10)/10, mode='valid')
plt.plot(smoothed, label='Custom', color='blue')
plt.title("Average Agent Distance Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Avg Agent Distance")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('custom_algorithm_metrics.png')
plt.show()

# Print Final Metrics
print("\nFinal Metrics (Last 50 Episodes):")
print(f"Average Shaped Reward: {np.mean(rewards_history[-50:]):.2f} ± {np.std(rewards_history[-50:]):.2f}")
print(f"Average Raw Reward: {np.mean(raw_rewards_history[-50:]):.2f} ± {np.std(raw_rewards_history[-50:]):.2f}")
print(f"Average Collisions (est): {np.mean(collision_history[-50:]):.2f} ± {np.std(collision_history[-50:]):.2f}")
print(f"Average Agent Distance: {np.mean(agent_distances_history[-50:]):.2f} ± {np.std(agent_distances_history[-50:]):.2f}")
print(f"Average Min Landmark Distance: {np.mean(landmark_dist_history[-50:]):.2f} ± {np.std(landmark_dist_history[-50:]):.2f}")
print(f"Average Landmarks Covered: {np.mean(landmarks_covered_history[-50:]):.2f} ± {np.std(landmarks_covered_history[-50:]):.2f}")