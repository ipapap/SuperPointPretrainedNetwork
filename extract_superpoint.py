import torch
import cv2
from demo_superpoint import SuperPointFrontend
import numpy as np
def get_points(img,conf_thresh=0.015,cuda=1,nn_thresh=0.7,nms_dist=4):
  weights_path='/mnt/sdd1/OneDrive/phd/dev/SuperPointPretrainedNetwork/superpoint_v1.pth'
  imgf=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
  imgf=imgf.astype(np.float32)
  torch.cuda.empty_cache()
  fe = SuperPointFrontend(weights_path=weights_path,
                          nms_dist=nms_dist,
                          conf_thresh=conf_thresh,
                          nn_thresh=nn_thresh,
                          cuda=cuda)
  pts, desc, heatmap = fe.run(imgf)
  return( pts, desc, heatmap)
