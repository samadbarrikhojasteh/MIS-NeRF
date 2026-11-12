import json
import os
from typing import Dict, List
import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torchvision
import cv2
dir= "data/liver/liver_yesfall/output_colmap"
dir_mask= os.path.join(dir,"mask")
dir_image= os.path.join(dir,"images_2")
img_torch=[]
if os.path.exists(dir_mask): print("You had the mask file already!")
else: os.mkdir(dir_mask) and print("The mask file has been created!!")
n= sorted(os.listdir(dir_image))
for i in range(len(n)):
    v= os.path.join(dir_image,n[i])
    pil_image = Image.open(v)
    image = (np.array(pil_image)).astype("uint8") # shape is (h, w) or (h, w, 3 or 4)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    gray_cv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255
    # gray_cv= gray_cv[:,:,None]
    mask= np.where(gray_cv> 0.8, 0.0, 1.0)
    # mask = np.where(gray3 > 0.8, 0.0, image)
    cv2.imwrite(os.path.join(dir_mask, n[i]), mask*255)






    # #mask= np.where(image/255 > 0.8, 0.0, 1.0)
    # mask = np.where(image > 0.8)
    # #mask_image= np.where(image > 0.8, 0.0, image/255)
    # x= mask[0].astype(int)
    # y= mask[1].astype(int)
    # image[x,y,:]= 500.0
    # image2= np.where(image== 500.0, 0.0, 1.0)
    # cv2.imwrite(os.path.join(dir_mask, n[i]), image2*255)



#     gray = np.mean(image, 2)/255
#     # gray_cv = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)/255
#     mask = np.where(gray > 0.8, 0.0, 1.0)
#     #rgb= np.stack((image[:,:,0]*0.299, image[:,:,1]*0.587, image[:,:,2]*0.114),2)
#     cv2.imwrite(os.path.join(dir_mask,n[i]), mask*255)
#     #sam= cv2.imread(os.path.join(dir_mask,n[i]))
#     # img= torch.from_numpy(mask.astype("float32"))
#     # # Apply mask
#     # # img= torch.where(img>0.6, 0, 1.0)
#     # img= img.view(img.shape[0], img.shape[1], 1)
#     # img= torch.permute(img,(2,0,1))
#     # torchvision.utils.save_image(img,os.path.join(dir_mask,n[i]))
#     img_torch.append(img)
# #all= torch.stack(img_torch)
#






#Get camera positions from txt file and convert to nerfstuduio format
dir= "pose.txt"
sam= np.loadtxt(dir,delimiter=",")
posss=[]
for o in range(0,len(sam)):
    poss = []
    for i in range(4):
        pos1 = sam[o][0+i]
        pos2 = sam[o][4+i]
        pos3 = sam[o][8+i]
        pos4 = sam[o][12+i]
        poss_np = np.stack((pos1, pos2, pos3, pos4)).tolist()
        poss.append(poss_np)
    posss.append(poss)





jsont= json.load(open("transforms.json"))
total=[]
for i in range(0, len(jsont["frames"])):
    img_name = jsont["frames"][i]["file_path"]
    num_img= int((os.path.split(img_name)[1].split("."))[0].split("_")[1])
    pos = jsont["frames"][i]["transform_matrix"]
    pos= posss[num_img-1]
    total.append(pos)
#
# with open("transforms.json", "w", encoding="UTF-8") as file:
#     json.dump(pos, file)