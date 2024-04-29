from torch.utils.data import Dataset
import torch
import os
import torchvision.transforms as transform
from PIL import Image

class myDataset(Dataset):
    def __init__(self,img_path,annot_path,img_size) -> None:
        super().__init__()
        self.imgs_path = img_path
        self.annots_path = annot_path
        self.images = os.listdir(self.imgs_path)
        self.annots = os.listdir(self.annots_path)
        self.img_size = img_size
        self.transforms = transform.Compose( transform.ToTensor() , 
                                            transform.Resize((img_size,img_size)) )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img = os.join(self.imgs_path,self.images[index])
        annot = os.join(self.annots_path,self.annots[index])
        img = Image.open(img).convert("RGB")
        img = self.transforms(img)
        height , width = img.shape[1:]
        top_left_x,top_left_y,bottom_right_x,bottom_right_y = annot[2],annot[0],annot[3],annot[1]
        annots = torch.tensor([float(top_left_x/width),float(top_left_y/height),float(bottom_right_x/width),float(bottom_right_y/height)])
        return img,annots
