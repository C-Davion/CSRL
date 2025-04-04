import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#adapted from the implementation of the paper



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Actor(nn.Module):
    def __init__(self, state_space_dim, action_state_dim, hidden_layer_param, max_action):
        super(Actor, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(state_space_dim, hidden_layer_param[0])])  

        for i in range(len(hidden_layer_param) - 1):
            self.layers.append(nn.Linear(hidden_layer_param[i], hidden_layer_param[i + 1]))

        self.layers.append(nn.Linear(hidden_layer_param[-1], action_state_dim))

        self.max_action = max_action

    def forward(self, state):
        x = state
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        x = self.layers[-1](x)
        return self.max_action * torch.tanh(x)
    


class Critic(nn.Module):
    def __init__(self, state_space_dim, action_state_dim, hidden_layer_param):
        super(Critic, self).__init__()
        self.Q1layers = nn.ModuleList([nn.Linear(state_space_dim + action_state_dim, hidden_layer_param[0])])  #Since it's Q(s,a), we will be stacking those
        self.Q2layers = nn.ModuleList([nn.Linear(state_space_dim + action_state_dim, hidden_layer_param[0])])  

        for i in range(len(hidden_layer_param) - 1):
            self.Q1layers.append(nn.Linear(hidden_layer_param[i], hidden_layer_param[i + 1])) #Q1 aand Q2 have the same architecture
            self.Q2layers.append(nn.Linear(hidden_layer_param[i], hidden_layer_param[i + 1]))

        self.Q1layers.append(nn.Linear(hidden_layer_param[-1], 1))
        self.Q2layers.append(nn.Linear(hidden_layer_param[-1], 1))

    def forward(self, state, action):
        st_and_ac1 = torch.cat([state, action], 1)
        st_and_ac2 = copy.deepcopy(st_and_ac1)

        for i in range(len(self.Q1layers) - 1):
            st_and_ac1, st_and_ac2 = self.Q1layers[i](st_and_ac1), self.Q2layers[i](st_and_ac2)
            st_and_ac1, st_and_ac2 = F.relu(st_and_ac1), F.relu(st_and_ac2)

        st_and_ac1, st_and_ac2 = self.Q1layers[-1](st_and_ac1), self.Q2layers[-1](st_and_ac2)
        return st_and_ac2, st_and_ac1

    def Q1(self, state, action):
        st_and_ac = torch.cat([state, action], 1)
        for i in range(len(self.Q1layers) - 1):
            st_and_ac = self.Q1layers[i](st_and_ac)
            st_and_ac = F.relu(st_and_ac)
        st_and_ac = self.Q1layers[-1](st_and_ac)
        return st_and_ac


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
        hidden_layer_param=[256,256],
		max_action=1,
		discount=0.99,
		tau=0.005, #θ'<-τθ+(1-τ)θ' 
		policy_noise=0.2,
		noise_clip=0.5,
		d=2 #the timestep neede before updating the actors, for delayed policy update
	):

		self.actor = Actor(state_dim, action_dim,hidden_layer_param, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor) #target network
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4) #TD3 paper implementationd details

		self.critic = Critic(state_dim, action_dim,hidden_layer_param).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.d = d

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			)#.clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.d == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

