import numpy as np
class RewardManager():
    """Computes and returns rewards based on states and actions."""
    def __init__(self):
        pass

    def get_reward(self, state, action):
        """Returns the reward as a dictionary. You can include different sub-rewards in the
        dictionary for plotting/logging purposes, but only the 'reward' key is used for the
        actual RL algorithm, which is generated from the sum of all other rewards."""
        reward_dict = {}
        # Your code here
        
        # ['collision', 'speed', 'waypoint_dist', 'waypoint_angle', 'command', 'route_dist', 'route_angle', 'lane_dist', 'lane_angle', 'is_junction', 'hazard', 'hazard_dist', 'hazard_coords', 'tl_state', 'tl_dist', 'optimal_speed', 'measurements', 'rgb', 'gps', 'imu']
        # {'throttle': 0.4816156029701233, 'brake': 0.0, 'steer': -0.0468420535326004}
        # command: LEFT = 0, RIGHT = 1, STRAIGHT = 2, LANEFOLLOW = 3
        # steer -1 to 1 
        # route angle x to -x
        
        max_speed=5.0
        speed_reward = 2.0*min(state["speed"],max_speed) #5-abs(state["speed"] - 5) #
        
        if state["speed"] > max_speed:
                speed_reward += max_speed-state["speed"]
        reward_dict["speed"] = speed_reward
        
        if state["tl_state"] == 1:
                tl = -state['speed']
                tl += 2.0*action['brake']
                tl += -2.0*action['throttle']
                reward_dict["tl"] = -5.0*tl
        
        if state["collision"]:
                reward_dict["collision"] = -400.0
        
        max_dist = 10.0
        max_dist_lane = 4.0
        centering_factor = max(1.0 - state['route_dist'] / max_dist, 0.0)
        centering_factor_lane = max(1.0 - state['lane_dist'] / max_dist, 0.0)
        
        reward_dict['centering'] = 2.0* centering_factor
        reward_dict['centering_lane'] = 2.0* centering_factor_lane
        
        steer_retrieve = 0.0
        if state['command'] == 2 or state['command'] == 3:
                angle_factor = max(1.0 - abs(state['route_angle'] / np.deg2rad(20.0)), -1.0)
                reward_dict['angle'] = 5.0 * angle_factor
                
                if state['route_angle'] > 0:
                        if action['steer'] > 0:
                                steer_retrieve += 5.0
                        else:
                                steer_retrieve += 5.0*action['steer']
                else:
                        if action['steer'] < 0:
                                steer_retrieve += 5.0
                        else:
                                steer_retrieve += -5.0*action['steer']
        reward_dict['steer_retrieve'] = steer_retrieve
        
        steer = 0.0
        if state['command'] == 0:
                if action['steer'] > 0:
                        steer += -10
                else:
                        steer += -10*action['steer'] #is -ve
        elif state['command'] == 1:
                if action['steer'] > 0:
                        steer += 10*action['steer']
                else:
                        steer += -10
        reward_dict['steer'] = steer
        

        # Your code here
        reward = 0.0
        for val in reward_dict.values():
            reward += val
        reward_dict["reward"] = reward
        return reward_dict
