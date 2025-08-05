

import torch
import torch.nn as nn



class YOLO(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = nn.Sequential(
    
    nn.Conv2d(3, 64, 7, stride=2),
    nn.MaxPool2d(2, stride=2),

    nn.Conv2d(64, 192, 3),
    nn.MaxPool2d(2, stride=2),
    
    nn.Conv2d(192, 128, 1),
    nn.Conv2d(128, 256, 3),
    nn.Conv2d(256, 256, 1),
    nn.Conv2d(256, 512, 3),
    nn.MaxPool2d(2, stride=2),
    
    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3),
   
    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3),
    
    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3),

    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3),

    nn.Conv2d(512, 512, 1),
    nn.Conv2d(512, 1024, 3),
    nn.MaxPool2d(2, stride=2),
    # 1 x 1024 x 7 x 7
    nn.Conv2d(1024, 512, 1), # 1 x 512, 7, 7
    
    nn.Conv2d(512, 1024, 3),  # 1, 1024, 5, 5
    
    nn.Conv2d(1024, 512, 1),
    nn.Conv2d(512, 1024, 3),  # 1, 1024, 3, 3

    nn.Conv2d(1024, 1024, 3),  # 1, 1024, 1, 1
    # We have issue of mismatch between the shapes of paper and default torch conv2d operations!
    '''
    nn.Conv2d(1024, 1024, 3, stride=2),
    nn.Conv2d(1024, 1024, 3),
    nn.Conv2d(1024, 1024, 3),
    #nn.Flatten(),
    )'''
  def forward(self, x): return self.backbone(x)


 
model = YOLO()
x = torch.Tensor(1, 3, 448, 448)
out = model(x)
print(out.shape)





