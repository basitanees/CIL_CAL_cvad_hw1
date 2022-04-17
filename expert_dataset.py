import torch
import os
from PIL import Image
import json
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root): #/home/aanees20/My Data/Comp 523/Homework 1/expert_data_subset/val
        self.data_root = data_root
        self.measurements_root = os.path.join(data_root, "measurements")
        self.images_root = os.path.join(data_root, "rgb")
        self.measurements = sorted([os.path.join(self.measurements_root,f) for f in os.listdir(self.measurements_root)])
        self.images = sorted([os.path.join(self.images_root,f) for f in os.listdir(self.images_root)])
        self.transforms = transforms.Compose([
                                        transforms.Resize(224),
#                                         transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

        # Your code here
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        image = Image.open(self.images[index])
        image = image.convert('RGB')
        image = self.transforms(image)
        with open(self.measurements[index], 'r') as j:
            stuff = json.loads(j.read())
        measurements = {}
        measurements['speed'] = torch.tensor(stuff['speed']).unsqueeze(-1)
        measurements['labels'] = torch.tensor([stuff['steer'], stuff['throttle'], stuff['brake']]) #, stuff['speed']
        measurements['affordance_cont'] = torch.tensor([stuff['route_angle'], stuff['lane_dist'], stuff['tl_dist']]) #, stuff['speed']
        measurements['tl_state'] = torch.tensor(stuff['tl_state']).unsqueeze(-1)
        measurements['rgb'] = image
        for key, val in measurements.items():
            measurements[key] = val.to(torch.device("cuda")).float()
        measurements['command'] = torch.tensor(stuff['command']).long()
        return measurements
