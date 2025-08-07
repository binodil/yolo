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
    org_h, org_w = img.shape[:2]
    # preprocessing logic
    img = cv2.resize(img, (448, 448))
    img = img / 255.0
    img = torch.FloatTensor(img)
    img = img.permute(2, 0, 1)
    
    boxes = self.labels[idx]
    label = torch.zeros((S, S, (C + 5*B)))
    # https://github.com/aladdinpersson/Machine-Learning-Collection/issues/140
    for box in boxes:
      # we need to present it
      x_center = int((box[1] * org_w) * 448/org_w)
      y_center = int((box[2] * org_h) * 448 / org_h)
      width = int((box[3] * org_w) * 448 / org_w)
      height = int((box[4] * org_h) * 448 / org_h)
      class_i = box[0]
      S_grid_size = 448//S
      
      # x, y is normalized relative to the grid (not whole image) our grid is (64x64).
      # width and height is normalized relative to the whole image.
      y_idx = y_center // S_grid_size
      x_idx = x_center // S_grid_size
      
      x_wrt_grid = (x_center - x_idx*S_grid_size) / S_grid_size
      y_wrt_grid = (y_center - y_idx*S_grid_size) / S_grid_size

      if label[y_idx, x_idx, :C].sum() == 0:
        label[y_idx, x_idx, class_i] = 1
        label[y_idx, x_idx, C:(C+5)] = torch.from_numpy(np.array([x_wrt_grid, y_wrt_grid, box[3], box[4], 1]))
      else:
        label[y_idx, x_idx, (C+5):(C+5*B)] = torch.from_numpy(np.array([x_wrt_grid, y_wrt_grid, box[3], box[4], 1]))

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
  BATCH_SIZE = 2
  LEARNING_RATE = 0.00001
  EPOCH_NUM = 10

  #---Loss hyperparams
  theta_coord = 5
  theta_noobj = .5

  path = pathlib.Path("/mnt/d/pascal/")
  train_dataloader = DataLoader(Pascal(path, "train.csv"), batch_size=BATCH_SIZE)
  #test_dataloader = DataLoader(Pascal(path, "test.csv"))
  
  model = yolo.YOLO(S=7, B=2, C=20)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  
  for epoch in range(EPOCH_NUM):
    for i, batch in enumerate(train_dataloader):
    #for i, batch in enumerate([(torch.randn(BATCH_SIZE, 3, 448, 448), torch.randn(BATCH_SIZE, 7, 7, 30))]):
      images, label_true = batch
      print(images.shape)
      optimizer.zero_grad()
      label_hat = model(images)
      label_hat = label_hat.reshape(BATCH_SIZE, S, S, C+5*B)
      print(label_true.shape)
      # calculate loss between y_hat and y_true
      # Loss on x and y of object
      total_loss = -torch.inf
      for i in range(S):
        for j in range(S):
          mse_xy = -torch.inf
          mse_wh = -torch.inf
          mse_class_cond = -torch.inf
          mse_no_class_cond = -torch.inf
          class_cond_true = label_true[:, i, j, :C]
          class_cond_hat = label_hat[:, i, j, :C]
 
          for b in range(B):
            out = label_hat[:, i, j, C+b*5:C+(b+1)*5]
            x_hat = out[:, 0]
            y_hat = out[:, 1]
            w_hat = out[:, 2]
            h_hat = out[:, 3]
            score = out[:, 4]

            out = label_true[:, i, j, (C + b*5):C+(b+1)*5]
            x_true = out[:, 0]
            y_true = out[:, 1]
            w_true = out[:, 2]
            h_true = out[:, 3]
            score_true = out[:, 4]
            #print(score_true)            
            # we have nan value in mse_wh formula. We shall fix it!
            for i, t in enumerate([w_hat, h_hat, w_true, h_true]):
              if t.nansum() > 0:
                  print(i, t)
            mse_xy = (((x_true - x_hat)**2 + (y_true - y_hat)**2) * score_true).sum()

            mse_wh = ((torch.sqrt(w_true) - torch.sqrt(w_hat))**2 + (torch.sqrt(h_true) - torch.sqrt(h_hat))**2 * score_true).sum()
            print(mse_wh)
            print(mse_xy)
            # if object appears, can use the score and if it is non zero then use?
            # How to know if object appears in cell [i, j].

            #if score_true == 1:
            #  print("Yes score is one")
              #mse_xy += (x_true - x_hat)**2 + (y_true - y_hat)**2
              #mse_wh += (w_true**(1/2) - w_hat**(1/2))**2 + (h_true**(1/2) - h_hat**(1/2))**2
              #print("mse_xy:", mse_xy)
              #print("mse_wh:", mse_wh)

              #mse_class_cond += ((class_cond_true - class_cond_hat)**2).sum()
              #mse_no_class_cond += ((class_cond_true - class_cond_hat)**2).sum()
            
            # for each cell grid if object exists do sum of (p_i(c) - p_i_hat(c))**2




      # do backprop
      # optimizer.step()
      break
    break



    












