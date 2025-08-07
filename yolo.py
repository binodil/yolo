# implementation of YOLO paper by Joseph Redmon et al. (2016)
# credits to https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/object_detection/YOLO/model.py
import torch
import torch.nn as nn


# Padding is not defined in the paper. We are doing reverse engineering.
#----------------------------------------------------------------------

class YOLO(nn.Module):
  def __init__(self, S:int, C:int, B:int):
    # S: split size
    # C: num classes
    # B: num of boxes

    super().__init__()
    self.backbone = nn.Sequential(
    
    nn.Conv2d(3, 64, 7, stride=2, padding=1),
    nn.MaxPool2d(2, stride=2, padding=1),  # expected size is 64x112x112
    
    
    nn.Conv2d(64, 192, 3, padding=1),
    nn.MaxPool2d(2, stride=2),  #192x56x56

    nn.Conv2d(192, 128, 1, padding=1),
    nn.Conv2d(128, 256, 3, padding=1),
    nn.Conv2d(256, 256, 1),
    nn.Conv2d(256, 512, 3),
    nn.MaxPool2d(2, stride=2),  # 512x28x28
    

    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3, padding=1),
   
    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3, padding=1),
    
    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3, padding=1),

    nn.Conv2d(512, 256, 1),
    nn.Conv2d(256, 512, 3, padding=1),

    nn.Conv2d(512, 512, 1),
    nn.Conv2d(512, 1024, 3, padding=1),
    nn.MaxPool2d(2, stride=2),  # 1024x14x14


    nn.Conv2d(1024, 512, 1),
    nn.Conv2d(512, 1024, 3, padding=1),
    nn.Conv2d(1024, 512, 1),
    nn.Conv2d(512, 1024, 3, padding=1),
    nn.Conv2d(1024, 1024, 3, padding=1),
    nn.Conv2d(1024, 1024, 3, stride=2, padding=1),  # 1024x7x7

    nn.Conv2d(1024, 1024, 3, padding=1),
    nn.Conv2d(1024, 1024, 3, padding=1),  # 1024x7x7

    nn.Flatten(),
    nn.Linear(1024*S*S, 4096),
    nn.ReLU(),
    # Dropout?
    # LeakyReLU? BatchNorm?
    nn.Linear(4096, S*S*(C+5*B)),
    )
  def forward(self, x): return self.backbone(x)


if __name__ == "__main__": 
  model = YOLO(S=7, B=2, C=20)  # same as in paper
  x = torch.randn(1, 3, 448, 448)
  out = model(x)
  print(out.shape)
  print(out.reshape(1, 7, 7, 30).shape)




