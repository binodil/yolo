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

def intersect_over_union(truth_bbox, pred_bbox, img_size, S):
  # img_size = 448; We need because bbox are normalized!
  # S to unnormalize x_center and y_center.
  def convert_bbox(bbox):
    x, y, w, h = bbox.T
    w = w * img_size
    h = h * img_size
    x = x * img_size/S  # X w.r.t. correcponding grid cell
    y = y * img_size/S
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return x1, y1, x2, y2
  x1, y1, x2, y2 = convert_bbox(truth_bbox)
  x3, y3, x4, y4 = convert_bbox(pred_bbox)
  x1_inter = torch.max(x1, x3)
  y1_inter = torch.max(y1, y3)
  x2_inter = torch.min(x2, x4)
  y2_inter = torch.min(y2, y4)
  intersect = (x2_inter - x1_inter) * (y2_inter - y1_inter)
  intersect[(x1_inter>x2_inter) & (y1_inter>y2_inter)] = 0.0
  iou = intersect / (((x2-x1)*(y2-y1)) + ((x4-x3)*(y4-y3)) - intersect)
  return iou
  

if __name__ == "__main__":
  #bb1 = torch.Tensor([[0.5, 0.5, 0.6, 0.7], [0.5, 0.5, 0.6, 0.7], [0.9, 0.9, 0.1, 0.1], [0.5, 0.5, 0.6, 0.7], [0.9, 0.9, 0.001, 0.001]])
  #bb2 = torch.Tensor([[0.5, 0.5, 0.3, 0.7], [0.5, 0.5, 0.3, 0.35], [0.9, 0.9, 0.1, 0.1], [0, 0, 0, 0], [0.1, 0.1, 0.001, 0.001]])  
  #intersect_over_union(bb1, bb2, img_size=448, S=7)
  #import sys; sys.exit(0)
   
  import time
  import yolo

  BATCH_SIZE = 64
  LEARNING_RATE = 0.00003
  EPOCH_NUM = 100

  #---Loss hyperparams
  theta_coord = 5
  theta_noobj = .5
  device = torch.device("mps")

  # path = pathlib.Path("/mnt/d/pascal/")  # WSL.exe in Windows
  path = pathlib.Path("/Users/sardor/fun/yolo/pascalvoc-yolo")  # MacOS
  train_dataloader = DataLoader(Pascal(path, "train.csv"), batch_size=BATCH_SIZE)
  #test_dataloader = DataLoader(Pascal(path, "test.csv"))
  
  model = yolo.YOLO(S=7, B=2, C=20)
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  for epoch in range(EPOCH_NUM):
    strt_epoch = time.perf_counter()  
    for btch_i, batch in enumerate(train_dataloader):
      strt_batch = time.perf_counter()
      images, label_true = batch
      images = images.to(device)
      label_true = label_true.to(device)
      optimizer.zero_grad()
      label_hat = model(images)
      label_hat = label_hat.reshape(BATCH_SIZE, S, S, C+5*B)
      mse_xy = 0
      mse_wh = 0
      mse_conf_score = 0
      mse_no_conf_score = 0
      mse_class_loss = 0
      label_true = label_true.flatten(1, 2) # [batch, 49, 30]
      label_hat = label_hat.flatten(1, 2)  # [batch, 49, 30]
      
      for i in range(49):
        class_cond_true = label_true[:, i, :C]
        class_cond_hat = label_hat[:, i, :C]

        for b in range(B):
          out = label_hat[:, i, C+b*5:C+(b+1)*5]
          x_hat = out[:, 0]
          y_hat = out[:, 1]
          w_hat = out[:, 2]
          h_hat = out[:, 3]
          score_hat = out[:, 4]
          out_true = label_true[:, i, C + b*5:C+(b+1)*5]
          x_true = out_true[:, 0]
          y_true = out_true[:, 1]
          w_true = out_true[:, 2]
          h_true = out_true[:, 3]
          obj_exists = out_true[:, 4]
          score_true = intersect_over_union(out_true[:, :4], out[:, :4], img_size=448, S=S)
          mse_xy += (((x_true - x_hat)**2 + (y_true - y_hat)**2) * obj_exists)  #.sum()
          mse_wh += ((torch.sqrt(w_true) - w_hat.sign()*torch.sqrt(w_hat.abs()))**2 + (torch.sqrt(h_true) - h_hat.sign()*torch.sqrt(h_hat.abs()))**2 * score_true)  #.sum()
          mse_conf_score += obj_exists * (score_true - score_hat)**2
          mse_no_conf_score += (1-obj_exists)*(score_true-score_hat)**2
      # here we need 5th loss
        mse_class_loss += ((class_cond_true - class_cond_hat)**2 * class_cond_true).sum(1) #.sum()
    
      #print(mse_class_loss)
      # class loss
        # for each cell grid if object exists do sum of (p_i(c) - p_i_hat(c))**2
      #print(f"{mse_xy=}, {mse_wh=}, {mse_conf_score=}, {mse_no_conf_score=}, {mse_class_loss=}")
      total_loss = theta_coord * mse_xy + theta_coord * mse_wh + mse_conf_score + theta_noobj * mse_no_conf_score + mse_class_loss
      #print(total_loss)
      total_loss = torch.mean(total_loss)
      total_loss.backward()
      optimizer.step()
      fnsh_batch = time.perf_counter()
      avrg_time = fnsh_batch - strt_batch
      print(f"{epoch} epoch | total loss: {total_loss.item():.2f} | dur (s): {avrg_time:.2f} | {btch_i}/{len(train_dataloader)} | Remaining: {avrg_time*(len(train_dataloader) - i)}")
      #import pdb; pdb.set_trace()
    fnsh_epoch = time.perf_counter()
      # do backprop
      # optimizer.step()


