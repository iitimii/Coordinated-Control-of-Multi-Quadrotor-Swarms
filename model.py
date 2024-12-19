import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class DroneActor(nn.Module):
    def __init__(
        self,
        image_shape=(48, 64, 4),
        kinematics_dim=12,
        action_dim=4,
        hidden_dim=256
    ):
        super().__init__()
        
        # Image processing
        self.conv_net = nn.Sequential(
            nn.Conv2d(image_shape[2], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        test_tensor = torch.zeros(1, image_shape[2], image_shape[0], image_shape[1])
        conv_out_size = self.conv_net(test_tensor).shape[1]
        
        # Kinematics processing
        self.kinematics_net = nn.Sequential(
            nn.Linear(kinematics_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(conv_out_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, image, kinematics):
        # Process image: (B, C, H, W)
        image = image.permute(0, 3, 1, 2).float() / 255.0
        image_features = self.conv_net(image)
        
        # Process kinematics
        kinematics_features = self.kinematics_net(kinematics)
        
        # Combine features
        combined = torch.cat([image_features, kinematics_features], dim=1)
        features = self.combined_net(combined)
        
        # Get action distribution parameters
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample_action(self, image, kinematics):
        mean, log_std = self.forward(image, kinematics)
        std = log_std.exp()
        distribution = Normal(mean, std)
        action = distribution.rsample()
        return action
    
    def evaluate_actions(self, image, kinematics, actions):
        mean, log_std = self.forward(image, kinematics)
        std = log_std.exp()
        distribution = Normal(mean, std)
        log_probs = distribution.log_prob(actions).sum(-1, keepdim=True)
        entropy = distribution.entropy().sum(-1, keepdim=True)
        return log_probs, entropy

class DroneCritic(nn.Module):
    def __init__(
        self,
        image_shape=(48, 64, 4),
        kinematics_dim=12,
        hidden_dim=256
    ):
        super().__init__()
        
        # Image processing
        self.conv_net = nn.Sequential(
            nn.Conv2d(image_shape[2], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate conv output size
        test_tensor = torch.zeros(1, image_shape[2], image_shape[0], image_shape[1])
        conv_out_size = self.conv_net(test_tensor).shape[1]
        
        # Kinematics processing
        self.kinematics_net = nn.Sequential(
            nn.Linear(kinematics_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(conv_out_size + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, image, kinematics):
        # Process image
        image = image.permute(0, 3, 1, 2).float() / 255.0
        image_features = self.conv_net(image)
        
        # Process kinematics
        kinematics_features = self.kinematics_net(kinematics)
        
        # Combine features
        combined = torch.cat([image_features, kinematics_features], dim=1)
        value = self.combined_net(combined)
        
        return value

# Example usage with PPO
class DronePPO:
    def __init__(
        self,
        image_shape=(48, 64, 4),
        kinematics_dim=12,
        action_dim=4,
        hidden_dim=256,
        lr=3e-4
    ):
        self.actor = DroneActor(image_shape, kinematics_dim, action_dim, hidden_dim)
        self.critic = DroneCritic(image_shape, kinematics_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, obs):
        with torch.no_grad():
            image = torch.FloatTensor(obs['rgb']).unsqueeze(0)
            kinematics = torch.FloatTensor(obs['kinematics']).unsqueeze(0)
            action = self.actor.sample_action(image, kinematics)
        return action.squeeze(0).numpy()