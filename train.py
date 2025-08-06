import pathlib
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



# hyperparams
S = 7
C = 20
B = 2


class Pascal(Dataset):
  def __init__(self, path, csv_name):
    super().__init__()
    self.path = path
    self.image_paths = []
    self.labels = []

    with open(self.path/csv_name) as f:
      reader = csv.reader(f)
      for row in reader:
        self.image_paths.append(row[0])
        with open(self.path/'labels'/row[1]) as file:
          data = file.readlines()
          boxes = []
          for row in data:
            class_label, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x) for x in row.strip().split(" ")]
            boxes.append([class_label, x, y, w, h])

          self.labels.append(boxes)
    print("Loaded all annotations")

  def __len__(self): return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.path/"images"/self.image_paths[idx]
    img = cv2.imread(img_path)
    # preprocessing logic
    img = cv2.resize(img, (448, 448))
    img = img / 255.0
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1)
    
    boxes = self.labels[idx]
    # convert to the shape S*S*(C + 2*B)
    label = torch.zeros((S, S, (C + 5*B)))
    for box in boxes:
      # we need to present it
      x_center = int(box[1] * 448)
      y_center = int(box[2] * 448)
      width = int(box[3] * 448)
      height = int(box[4] * 448)
      class_i = box[0]
      S_grid_size = 448//S

      label[y_center//S_grid_size, x_center//S_grid_size, class_i] = 1
      if label[y_center//S_grid_size, x_center//S_grid_size, C:(C+B)].sum() == 0:
        label[y_center//S_grid_size, x_center//S_grid_size, C:(C+5)] = torch.from_numpy(np.array([box[1], box[2], box[3], box[4], 1,]))
      else:
        label[y_center//S_grid_size, x_center//S_grid_size, (C+5):(C+5*B)] = torch.from_numpy(np.array([box[1], box[2], box[3], box[4], 1,]))

    return img, label

# DataLoader

# train dataloader and test dataloader
# optimizer

# model

# train loop


# optimizer zero grad

# Loss function apply

# optimizer step (backprop)



if __name__ == "__main__":
  import yolo
  BATCH_SIZE = 8
  LEARNING_RATE = 0.001
  path = pathlib.Path("/mnt/d/pascal/")
  dataset = Pascal(path, "train.csv")
  train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
  test_dataloader = DataLoader(Pascal(path, "test.csv"))
  
  model = yolo.YOLO(S=7, B=2, C=20)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


  print(len(dataset))
  print(dataset.__getitem__(0))




    












