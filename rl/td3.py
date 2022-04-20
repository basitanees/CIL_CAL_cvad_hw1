import numpy as np
import torch
from func_timeout import FunctionTimedOut, func_timeout
from utils.rl_utils import generate_noisy_action_tensor

from .off_policy import BaseOffPolicy


class TD3(BaseOffPolicy):
    def _compute_q_loss(self, data):
        """Compute q loss for given batch of data."""
        # Your code here
        target_noise = self.config["target_policy_noise"]
        noise_clip = self.config["target_policy_max_noise"]
        act_limit = 1
        gamma = self.config["discount"]
        
        stored_features, stored_command, stored_action, stored_reward, stored_new_features, stored_new_command, stored_is_terminal = data
        
        stored_features = torch.stack(stored_features, dim=-1).cuda()
        action = stored_action.cuda()
        stored_new_features = torch.stack(stored_new_features, dim=-1).cuda()
        done = (stored_is_terminal*1.0).cuda()
        reward = stored_reward.cuda()
        
        q1 = self.q_nets[0](stored_features, action)
        q2 = self.q_nets[1](stored_features, action)
        
        with torch.no_grad():
            targ = self.target_policy(stored_new_features, [stored_new_command])

            epsilon = torch.randn_like(targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            action2 = targ + epsilon
            action2 = torch.clamp(action2, -act_limit, act_limit)

            q1_targ = self.target_q_nets[0](stored_new_features, action2)
            q2_targ = self.target_q_nets[1](stored_new_features, action2)
            q1_targ = torch.min(q1_targ, q2_targ)
            backup = reward + gamma * (1 - done) * q_targ
        
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def _compute_p_loss(self, data):
        """Compute policy loss for given batch of data."""
        # Your code here

        features = torch.stack(data[0], dim=-1).cuda()
        command = data[1]

        q1_pi = self.q_nets[0](features, self.policy(features,[command]))
        return -q1_pi.mean()


    def _extract_features(self, state):
        """Extract whatever features you wish to give as input to policy and q networks."""
        # Your code here
        ## ['collision', 'speed', 'waypoint_dist', 'waypoint_angle', 'command', 'route_dist', 'route_angle', 'lane_dist', 'lane_angle', 'is_junction', 'hazard', 'hazard_dist', 'hazard_coords', 'tl_state', 'tl_dist', 'optimal_speed', 'measurements', 'rgb', 'gps', 'imu']
        features = [state['speed'],state['route_dist'],state['route_angle'],state['tl_state'],state['tl_dist'],state['lane_dist'],state['lane_angle'],state['is_junction']*1.0]

        
        features = torch.tensor(features, dtype=torch.float).cuda()
        return features #.unsqueeze(0)

    def _take_step(self, state, action):
        try:
            action_dict = {
                "throttle": np.clip(action[0, 0].item(), 0, 1),
                "brake": abs(np.clip(action[0, 0].item(), -1, 0)),
                "steer": np.clip(action[0, 1].item(), -1, 1),
            }
            new_state, reward_dict, is_terminal = func_timeout(
                20, self.env.step, (action_dict,))
        except FunctionTimedOut:
            print("\nEnv.step did not return.")
            raise
        return new_state, reward_dict, is_terminal

    def _collect_data(self, state):
        """Take one step and put data into the replay buffer."""
        features = self._extract_features(state)
        if self.step >= self.config["exploration_steps"]:
            action = self.policy(features.unsqueeze(0), [state["command"]]) #, [state["command"]]
            #print(action.shape)
            action = generate_noisy_action_tensor(
                action, self.config["action_space"], self.config["policy_noise"], 1.0)
        else:
            action = self._explorer.generate_action(state)
        if self.step <= self.config["augment_steps"]:
            action = self._augmenter.augment_action(action, state)

        # Take step
        new_state, reward_dict, is_terminal = self._take_step(state, action)

        new_features = self._extract_features(state)
        #print(features.shape)

        # Prepare everything for storage
        stored_features = [f.detach().cpu().squeeze(0) for f in features]
        stored_command = state["command"]
        stored_action = action.detach().cpu().squeeze(0)
        stored_reward = torch.tensor([reward_dict["reward"]], dtype=torch.float)
        stored_new_features = [f.detach().cpu().squeeze(0) for f in new_features]
        stored_new_command = new_state["command"]
        stored_is_terminal = bool(is_terminal)

        self._replay_buffer.append(
            (stored_features, stored_command, stored_action, stored_reward,
             stored_new_features, stored_new_command, stored_is_terminal)
        )
        self._visualizer.visualize(new_state, stored_action, reward_dict)
        return reward_dict, new_state, is_terminal
