import torch.nn as nn


from models.cilrs import FC

class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor, self).__init__()
        n_output = 1000
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, num_classes=n_output)
        self.affordances = FC(params={'neurons': [n_output, 256, 256, 4],
                                       'dropouts': [0.0, 0.5,0.0],
                                       'end_layer': True})
        self.activation = torch.nn.Sigmoid()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
                                       

    def forward(self, img):
        x = self.resnet(img)
        affordances_output = self.affordances(x)
        cont_affordance = affordances_output[:,:3]
        disc_affordance = affordances_output[:,3:4]
        disc_affordance = self.activation(disc_affordance)
        return cont_affordance, disc_affordance
        
        
