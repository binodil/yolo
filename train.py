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
    # [B, 7*7, 4].T --> [4, B, 7*7]
    x, y, w, h = bbox.permute(2, 0, 1)
    # I can use Tensor.permute()
    w = w * img_size
    h = h * img_size
    x = x * img_size/S  # X w.r.t. correcponding grid cell
    y = y * img_size/S
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return x1, y1, x2, y2
  x1, y1, x2, y2 = convert_bbox(truth_bbox)  # x1: [B, 7*7], ..., y2: [B, 7*7]
  x3, y3, x4, y4 = convert_bbox(pred_bbox)   # x3: [B, 7*7], ..., y4: [B, 7*7]
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
  train_dataloader = DataLoader(Pascal(path, "train.csv"), batch_size=BATCH_SIZE, num_workers=2)
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
      class_cond_true = label_true[:, :, :C]
      class_cond_hat = label_hat[:, :, :C]
      x_true_1 = label_true[:, :, 20]
      y_true_1 = label_true[:, :, 21]
      w_true_1 = label_true[:, :, 22]
      h_true_1 = label_true[:, :, 23]
      score_true_1 = label_true[:, :, 24]
      x_true_2 = label_true[:, :, 25]
      y_true_2 = label_true[:, : , 26]
      w_true_2 = label_true[:, :, 27]
      h_true_2 = label_true[:, :, 28]
      score_true_2 = label_true[:, :, 29]

      x_hat_1 = label_hat[:, :, 20]
      y_hat_1 = label_hat[:, :, 21]
      w_hat_1 = label_hat[:, :, 22]
      h_hat_1 = label_hat[:, :, 23]
      score_hat_1 = label_hat[:, :, 24]
      x_hat_2 = label_hat[:, :, 25]
      y_hat_2 = label_hat[:, :, 26]
      w_hat_2 = label_hat[:, :, 27]
      h_hat_2 = label_hat[:, :, 28]
      score_hat_2 = label_hat[:, :, 29]
      # obj_exists is depricated. Use genuine variable <<score_true>>.
      mse_xy = ((x_true_1 - x_hat_1)**2 + (y_true_1 - y_hat_1)**2) * score_true_1 + ((x_true_2 - x_hat_2)**2 + (y_true_2 - y_hat_2)**2) * score_true_2
      mse_wh = ((torch.sqrt(w_true_1) - w_hat_1.sign() * torch.sqrt(w_hat_1.abs()))**2 + (torch.sqrt(h_true_1) - h_hat_1.sign()*torch.sqrt(h_hat_1.abs()))**2) * score_true_1 + ((torch.sqrt(w_true_2) - w_hat_2.sign() * torch.sqrt(w_hat_2.abs()))**2 + (torch.sqrt(h_true_2) - h_hat_2.sign()*torch.sqrt(h_hat_2.abs()))**2) * score_true_2
      #import pdb; pdb.set_trace()
      iou_score_1 = intersect_over_union(label_true[:, :, 20:24], label_hat[:, :, 20:24], img_size=448, S=S)
      iou_score_2 = intersect_over_union(label_true[:, :, 25:29], label_hat[:, :, 25:29], img_size=448, S=S)
      mse_conf_score = score_true_1 * (iou_score_1 - score_hat_1)**2 + score_true_2 * (iou_score_2 - score_hat_2)**2
      mse_no_conf_score = (1-score_true_1)*(iou_score_1 - score_hat_1)**2 + (1-score_true_2) * (iou_score_2 - score_hat_2)**2
      mse_class_loss = ((class_cond_true - class_cond_hat)**2 * class_cond_true).sum(2) #.sum()
      print(f"{mse_xy.sum().item()=}, {mse_wh.sum().item()=}, {mse_conf_score.sum().item()=}, {mse_no_conf_score.sum().item()=}, {mse_class_loss.sum().item()=}")
      total_loss = theta_coord * mse_xy.sum() + theta_coord * mse_wh.sum() + mse_conf_score.sum() + theta_noobj * mse_no_conf_score.sum() + mse_class_loss.sum()
      #print(total_loss)
      total_loss = torch.mean(total_loss)
      total_loss.backward()
      optimizer.step()
      fnsh_batch = time.perf_counter()
      avrg_time = fnsh_batch - strt_batch
      print(f"{epoch} epoch | total loss: {total_loss.item():.2f} | dur (s): {avrg_time:.2f} | {btch_i}/{len(train_dataloader)} | Remaining: {avrg_time*(len(train_dataloader) - btch_i)}")
    fnsh_epoch = time.perf_counter()
      # do backprop
      # optimizer.step()


