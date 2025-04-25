import re
import string
from tkinter import N
import torch 
import torch.nn as nn 
import argparse 

architecture_cong = [
    (7, 64, 2, 3),
    "M",  # Maxpooling
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # it ensures that  all items are certainly .  
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

    
# get arguments from commandline
def get_args():
    
    parser = argparse.ArgumentParser()
    
    # parser.add_argument()
    
    return parser


class CNNBlock(nn.Module):
    def __init__(self, in_chanels, out_chanel, **kwargs):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_chanels, out_chanel, bias= False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_chanel)
        self.act = nn.LeakyReLU(0.1)
        
        
    def forward(self, x):
        return self.act(self.batchnorm(self.conv(x)))
    
    

class Yolov1(nn.Module):
    def __init__(self, in_chanels = 3,**kwargs):
        super(Yolov1, self).__init__()
        
        self.architecture = architecture_cong
        
        self.in_chanels = in_chanels
        
        self.darkness = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
    
    def _create_conv_layers(self, architecture):
        layers = [] 
        in_chanels = self.in_chanels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_chanels, out_chanel= x[1],
                                   kernel_size = x[0], stride = x[2], padding = x[3])]
                
                in_chanels = x[1]
            elif type(x) == str:
                layers+= [nn.MaxPool2d(kernel_size=2 , stride=2)]
            elif type(x) == list:
                conv1 = x[0]    # tuple, 
                conv2 = x[1]    # tuple, 
                num_repeat = x[2]
                
                for _ in range(num_repeat):
                    
                    layers += [CNNBlock(in_chanels,conv1[1], kernel_size = conv1[0],
                        stride = conv1[2], padding = conv1[3])]
                    layers += [CNNBlock(conv1[1], conv2[1], kernel_size = conv2[0], 
                        stride = conv2[2], padding = conv2[3])]
                    in_chanels = conv2[1]
                    
        return nn.Sequential(*layers)
                
    def _create_fcs(self, split_size, number_boxes, num_classes):
        S, B, C  = split_size, number_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(), 
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.1),  
            nn.LeakyReLU(0.1), 
            nn.Linear(4096, S * S * (C + B * 5))
        )
                            
        
    def forward(self, x):
        x = self.darkness(x)
        
        return self.fcs(torch.flatten(x, start_dim=1))
    

def test_case():
    model = Yolov1(split_size=7, number_boxes=2, num_classes=20)

    x = torch.randn((2, 3, 448, 448))

    print(model(x).shape)
    
def main():
    test_case()
if __name__ == '__main__':
    

    
    main()