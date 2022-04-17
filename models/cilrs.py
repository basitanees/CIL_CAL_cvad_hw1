import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self, params=None, module_name='Default'
                 ):
        super(FC, self).__init__()

        self.layers = []

        for i in range(0, len(params['neurons']) -1):
            fc = nn.Linear(params['neurons'][i], params['neurons'][i+1])
            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)

            if i == len(params['neurons'])-2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class Join(nn.Module):

    def __init__(self, params=None, module_name='Default'):

        super(Join, self).__init__()
        self.after_process = params['after_process']
        self.mode = params['mode']

    def forward(self, x, m):
        # get only the speeds from measurement labels
        if self.mode == 'cat':
            j = torch.cat((x, m), 1)
        else:
            raise ValueError("Mode to join networks not found")
        return self.after_process(j)

class Branching(nn.Module):

    def __init__(self, branched_modules=None):
        super(Branching, self).__init__()
        self.branched_modules = nn.ModuleList(branched_modules)

    def forward(self, x):
        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(x))
        branches_outputs = torch.stack(branches_outputs, dim = 1)
        return branches_outputs

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS, self).__init__()
        n_output = 1000
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, num_classes=n_output)
        self.measurements = FC(params={'neurons': [1, 128, 128],
                                       'dropouts': [0.0, 0.0],
                                       'end_layer': False})
        self.join = Join(
                        params={'after_process':
                                     FC(params={'neurons': [128 + n_output, 512],
                                                'dropouts': [0.0],
                                                'end_layer': False}),
                                'mode': 'cat'
                                }
         )
        self.speed_branch = FC(params={'neurons': [n_output,256,256,1],
                                       'dropouts': [0.0, 0.5,0.0],
                                       'end_layer': True})
        
        # Create the fc vector separatedely
        branch_fc_vector = []
        for i in range(4):
            branch_fc_vector.append(FC(params={'neurons': [512, 256, 256] +
                                                         [len(['steer', 'throttle', 'brake'])],
                                               'dropouts':  [0.0, 0.5, 0.0],
                                               'end_layer': True}))

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, img, command):
        x = self.resnet(img)

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(command)
        """ Join measurements and perception"""
        j = self.join(x, m)
        branch_outputs = self.branches(j)
        speed_branch_output = self.speed_branch(x)

        return branch_outputs, speed_branch_output
