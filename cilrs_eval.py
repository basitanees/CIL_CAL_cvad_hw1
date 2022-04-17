import os

import yaml

from carla_env.env import Env
import carla
import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()
        self.transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])

    def load_agent(self):
        # Your code here
        path = "cilrs_model.ckpt"
        agent = torch.load(path).to(torch.device("cuda"))
        agent.eval()
        return agent

    def generate_action(self,rgb, command, speed):
        # Your code here
#         rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224,224))
        rgb = self.transforms(rgb).unsqueeze(0).to(torch.device("cuda")).float()
        # rgb = rgb[:,::-1,:,:].to(torch.device("cuda")).float()
        command = torch.tensor(command).long()
        speed = torch.tensor([speed]).unsqueeze(0).to(torch.device("cuda")).float()
        
        branches, speed_pr = self.agent(rgb, speed)
        action_pred = branches[torch.arange(branches.shape[0]),command,:]

        steer, throttle, brake = action_pred[:,0].item(), action_pred[:,1].item(), action_pred[:,2].item()
        steer = float(steer)
        throttle = float(throttle)
        brake = float(brake)
        steer = max(-1.0, min(steer, 1.0))
        throttle = max(0.0, min(throttle, 1.0))
        brake = max(0.0, min(brake, 1.0))
        return throttle, steer, brake

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, steer, brake = self.generate_action(rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            print(i)
            state, _, is_terminal = self.env.reset()
            with tqdm(range(5000), unit="batch") as tepoch:
                for j in tepoch:
#             for j in range(5000):
                    if is_terminal:
                        break
                    state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/50")

def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
