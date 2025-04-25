
import torch 
import torch.nn as nn 

## build a model from scratch

class CNNBlock(nn.Module):
    def __init__(self, in_chanels, out_chanels, **kwargs):
        super().__init__()
        self.conv2 =  nn.Conv2d(in_chanels , out_chanels, **kwargs)
    def forward(self):
        return self.conv2()


model_config =  [
    
                # Each tuple represents kernel_size, in_chanels, out_chanels, stride, padding  
                # config 7 * 7 times 64 , stride = 2
                (7, 3, 64, 2, 0)
                
]
class MyModel(nn.Module):
    def __init__(self,in_chanels, out_chanels, **kwargs):
        super().__init__( **kwargs)
        
        
        layers = []
        
        self.in_chanels = in_chanels
        for x in  model_config:
            
            layers += CNNBlock(in_chanels,  )
        
        


# model_v1 = CNNBlock(3, 6, kernel_size = 3, padding = "same")


# model_v2 = CNNBlock(6, 32, stride = 3 , kernel_size = 7, padding = None)
# print(model_v1)
# print(model_v2)