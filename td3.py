import os
import numpy as np
import torch 
import torch.nn.functional as F 
from model import Critic, GetTemplate, GetAction
from utils import get_reaction_mask, plot_learning_curve, postprocessing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, max_size, state_dim, t_dim, action_dim, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((max_size, state_dim))
        self.t_memory = np.zeros((max_size, t_dim))
        self.t_mask_memory = np.zeros((max_size, t_dim))
        self.action_memory = np.zeros((max_size, action_dim))
        self.reward_memory = np.zeros((max_size, ))
        self.next_state_memory = np.zeros((max_size, state_dim))
        self.done_memory = np.zeros((max_size, ), dtype=np.bool)

    def store(self, state, template, t_mask, action, reward, state_, done):
        mem_idx = self.mem_cnt % self.mem_size
        
        self.state_memory[mem_idx] = state['ecfp']
        self.t_memory[mem_idx] = template
        self.t_mask_memory[mem_idx] = t_mask
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.next_state_memory[mem_idx] = state_['ecfp']
        self.done_memory[mem_idx] = done
        
        self.mem_cnt += 1
        
    def sample_batch(self):
        mem_len = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)
        
        states  = self.state_memory[batch]
        templates = self.t_memory[batch]
        t_masks = self.t_mask_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        dones   = self.done_memory[batch]
        
        return states, templates, t_masks, actions, rewards, states_, dones
        
    def ready(self):
        return self.mem_cnt >= self.batch_size

class TD3:
    def __init__(self, env, state_dim, t_dim, action_dim, ckpt_dir, gamma=0.99,
                 tau=0.005, action_noise=0.1, policy_noise=0.2, policy_noise_clip=0.5,
                 delay_time=2, max_size=1e6, batch_size=256, max_episodes=20000,
                 max_action=8, temperature=1.0):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip 
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir
        self.max_episodes = max_episodes
        self.max_action = max_action
        self.temp = temperature
        self.temp_min = 0.1
        self.ANNEAL_RATE = 0.00006
        
        self.f_net = GetTemplate(state_dim=state_dim, t_dim=t_dim).to(device)
        self.pi_net = GetAction(state_dim=state_dim, t_dim=t_dim, act_dim=action_dim).to(device)
        self.critic1 = Critic(state_dim=state_dim, t_dim=t_dim, action_dim=action_dim).to(device)
        self.critic2 = Critic(state_dim=state_dim, t_dim=t_dim, action_dim=action_dim).to(device)
        
        self.target_f_net = GetTemplate(state_dim=state_dim, t_dim=t_dim).to(device)
        self.target_pi_net = GetAction(state_dim=state_dim, t_dim=t_dim, act_dim=action_dim).to(device)
        self.target_critic1 = Critic(state_dim=state_dim, t_dim=t_dim, action_dim=action_dim).to(device)
        self.target_critic2 = Critic(state_dim=state_dim, t_dim=t_dim, action_dim=action_dim).to(device)
        
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, t_dim=t_dim,
                                   action_dim=action_dim, batch_size=batch_size)
        
        self.update_network_parameters(tau=1.0)
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
            
        for f_params, target_f_params in zip(self.f_net.parameters(),
                                                 self.target_f_net.parameters()):
            target_f_params.data.copy_(tau * f_params + (1 - tau) * target_f_params)
            
        for pi_params, target_pi_params in zip(self.pi_net.parameters(),
                                                 self.target_pi_net.parameters()):
            target_pi_params.data.copy_(tau * pi_params + (1 - tau) * target_pi_params)
            
        for cri1_params, target_cri1_params in zip(self.critic1.parameters(),
                                                 self.target_critic1.parameters()):
            target_cri1_params.data.copy_(tau * cri1_params + (1 - tau) * target_cri1_params)
            
        for cri2_params, target_cri2_params in zip(self.critic2.parameters(),
                                                   self.target_critic2.parameters()):
            target_cri2_params.data.copy_(tau * cri2_params + (1 - tau) * target_cri2_params)
            
    def remember(self, state, template, t_mask, action, reward, state_, done):
        self.memory.store(state, template, t_mask, action, reward, state_, done)
        
    def select_action(self, temp, ob, train=True):
        self.f_net.eval()
        self.pi_net.eval()
        state = torch.tensor([ob['ecfp']], dtype=torch.float).to(device)
        T_mask = np.array(get_reaction_mask(ob['smi'], self.env.rxns))
        T_mask = torch.from_numpy(T_mask.astype(np.float32)).to(device)
        template = self.f_net.forward(state, T_mask, temp)
        action = self.pi_net.forward(state, template)
        
        if train:
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                                 dtype=torch.float).to(device)
            action = torch.clamp(action + noise, -1, 1)
        self.f_net.train()
        self.pi_net.train()
        
        return template.squeeze().detach().cpu().numpy(), action.squeeze().detach().cpu().numpy() 
    
    def update(self, temp):
        if not self.memory.ready():
            return
        
        states, templates, t_masks, actions, rewards, states_, dones = self.memory.sample_batch()
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        templates_tensor = torch.tensor(templates, dtype=torch.float).to(device)
        masks_tensor = torch.tensor(t_masks, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.float).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states_tensor = torch.tensor(states_, dtype=torch.float).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(device)
        
        with torch.no_grad():
            next_templates_tensor = self.target_f_net.forward(next_states_tensor, masks_tensor, temp)
            next_actions_tensor = self.target_pi_net.forward(next_states_tensor, next_templates_tensor) 
            action_noise = torch.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                        dtype=torch.float).to(device)
            # smooth noise
            action_noise = torch.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = torch.clamp(next_actions_tensor + action_noise, -1, 1)
            q1_ = self.target_critic1.forward(next_states_tensor, next_templates_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_templates_tensor, next_actions_tensor).view(-1)
            q1_[dones_tensor] = 0.0
            q2_[dones_tensor] = 0.0
            critic_val = torch.min(q1_, q2_) 
            target = rewards_tensor + self.gamma * critic_val
            
        q1 = self.critic1.forward(states_tensor, templates_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, templates_tensor, actions_tensor).view(-1)
        
        cri1_loss = F.mse_loss(q1, target.detach())
        cri2_loss = F.mse_loss(q2, target.detach())
        critic_loss = cri1_loss + cri2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()
        
        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return
        
        new_templates_tensor = self.f_net.forward(states_tensor, masks_tensor, temp)
        new_actions_tensor = self.pi_net.forward(states_tensor, new_templates_tensor)
        q1 = self.critic1.forward(states_tensor, new_templates_tensor, new_actions_tensor)
        actor_loss = - torch.mean(q1)
        self.pi_net.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        
        f_loss = actor_loss + F.mse_loss(templates_tensor, new_templates_tensor)
        self.f_net.optimizer.zero_grad()
        f_loss.backward()
        
        self.pi_net.optimizer.step()
        self.f_net.optimizer.step()
        
        self.update_network_parameters()
            
    def save(self, episode):
        if not os.path.exists(os.path.join(self.checkpoint_dir, 'TD3_model')):
            os.makedirs(os.path.join(self.checkpoint_dir, 'TD3_model'))
        if not os.path.exists(os.path.join(self.checkpoint_dir, 'TD3_target_model')):
            os.makedirs(os.path.join(self.checkpoint_dir, 'TD3_target_model'))
        
        self.f_net.save(self.checkpoint_dir + 'TD3_model/f_net_{}.pth'.format(episode))
        self.target_f_net.save(self.checkpoint_dir + 'TD3_target_model/target_f_net_{}.pth'.format(episode))
        
        self.pi_net.save(self.checkpoint_dir + 'TD3_model/pi_net_{}.pth'.format(episode))
        self.target_pi_net.save(self.checkpoint_dir + 'TD3_target_model/target_pi_net_{}.pth'.format(episode))
        
        self.critic1.save(self.checkpoint_dir + 'TD3_model/critic1_{}.pth'.format(episode))
        self.target_critic1.save(self.checkpoint_dir + 'TD3_target_model/target_critic1_{}.pth'.format(episode))
        
        self.critic2.save(self.checkpoint_dir + 'TD3_model/critic2_{}.pth'.format(episode))
        self.target_critic2.save(self.checkpoint_dir + 'TD3_target_model/target_critic2_{}.pth'.format(episode))
    
    def load(self, episode):        
        self.f_net.load(self.checkpoint_dir + 'TD3_model/f_net_{}.pth'.format(episode))
        self.target_f_net.load(self.checkpoint_dir + 'TD3_target_model/target_f_net_{}.pth'.format(episode))
        
        self.pi_net.load(self.checkpoint_dir + 'TD3_model/pi_net_{}.pth'.format(episode))
        self.target_pi_net.load(self.checkpoint_dir + 'TD3_target_model/target_pi_net_{}.pth'.format(episode))
        
        self.critic1.load(self.checkpoint_dir + 'TD3_model/critic1_{}.pth'.format(episode))
        self.target_critic1.load(self.checkpoint_dir + 'TD3_target_model/target_critic1_{}.pth'.format(episode))
        
        self.critic2.load(self.checkpoint_dir + 'TD3_model/critic2_{}.pth'.format(episode))
        self.target_critic2.load(self.checkpoint_dir + 'TD3_target_model/target_critic2_{}.pth'.format(episode))
    
    def train(self, timestamp):
        total_reward_history = []
        avg_reward_history = []
        info_his = []
        track_his = []
        
        main_dir = os.path.join('results', 'test-' + timestamp)
        if not os.path.exists(main_dir):
            os.makedirs(main_dir)
            
        for ep in range(self.max_episodes):
            smiles_list = self.env.smiles_list
            print('+' * 50 + 'Ep {}'.format(ep+1) + '+' * 50)
            total_reward = []
            done = False
            ob = self.env.reset()
            cur_temp = np.maximum(self.temp * np.exp(-self.ANNEAL_RATE * (ep + 1)), self.temp_min)
            infos = []
            while not done:
                print('select action')
                rxn_hot, action = self.select_action(cur_temp, ob, train=True)
                print('###', np.argmax(np.array(rxn_hot)))
                print('Env step')
                ob_, reward, done, info = self.env.step(ob, action, rxn_hot)
                if bool(info):
                    print(info)
                    total_reward.append(reward)
                    infos.append(info)
                t_mask = np.array(get_reaction_mask(ob_['smi'], self.env.rxns))
                self.remember(ob, rxn_hot, t_mask, action, reward, ob_, done)
                print('total_reward:', total_reward)
                ob = ob_
            if bool(infos):
                if self.env.smiles_list not in track_his:
                    track_his.append(self.env.smiles_list)
                    info_his.append(infos)
            if (ep + 1) % 8 == 0:
                self.update(cur_temp)
            if total_reward:
                total_reward_history.append(total_reward[-1])
            avg_reward = np.mean(total_reward_history[-100:])
            avg_reward_history.append(avg_reward)
            print('Ep {}, Reward {}, AvgReward {}'.format(ep + 1, total_reward, avg_reward))
            print()
            
            if (ep + 1) % 100 == 0:
                self.save(ep + 1)
        
        postprocessing(info_his, os.path.join(main_dir, 'syn_path.json.gz'))
        
        episodes = [i+1 for i in range(self.max_episodes)]
        plot_learning_curve(episodes, avg_reward_history, title='AvgReward', ylabel='reward',
                            figure_file=f'{main_dir}/reward_{timestamp}.png')
            
