import torch.nn as nn


from models.cilrs import FC, Branching

class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor, self).__init__()
        n_output = 1000
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, num_classes=n_output)
#         self.affordances = FC(params={'neurons': [n_output, 256, 256, 2],
#                                        'dropouts': [0.0, 0.5,0.0],
#                                        'end_layer': True})
        self.traffic_state = FC(params={'neurons': [n_output, 256, 256, 1],
                                       'dropouts': [0.0, 0.5,0.0],
                                       'end_layer': True})
        self.traffic_dist = FC(params={'neurons': [n_output, 256, 256, 1],
                                       'dropouts': [0.0, 0.5,0.0],
                                       'end_layer': True})
        # Create the fc vector separatedely
        branch_fc_vector = []
        for i in range(4):
            branch_fc_vector.append(FC(params={'neurons': [n_output, 256, 256] +
                                                         [len(['route_angle','lane_dist'])],
                                               'dropouts':  [0.0, 0.5, 0.0],
                                               'end_layer': True}))

        self.affordance_branches = Branching(branch_fc_vector)  # Here we set branching automatically
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def forward(self, img, command):
        x = self.resnet(img)
        affordances_output = self.affordance_branches(x)
        affordances_output = affordances_output[torch.arange(affordances_output.shape[0]),command,:]
        tl_state = self.traffic_state(x)
#         tl_state_off = (tl_state < 0)
        tl_dist = self.traffic_dist(x)
#         tl_dist[tl_state_off] = 45.0
        cont_affordance = torch.cat([affordances_output[:,0:2], tl_dist],dim=1)
        return cont_affordance, tl_state#.unsqueeze(-1)
