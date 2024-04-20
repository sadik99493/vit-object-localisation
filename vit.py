import torch
import torch.nn as nn
import os
import  numpy as np
import scipy.io
from PIL import  Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchsummary import summary
import math
import torch.optim as optim

path_images = "/vision _transformer/caltech-1011/101_ObjectCategories/101_ObjectCategories/airplanes"
path_annot = "/vision _transformer/caltech-1011/Annotations/Annotations/Airplanes_Side_2"

image_paths = []
annot_paths = []

for root, dirs, files in os.walk(path_images):
    for file in files:
        if file.endswith(".jpg"):  
            image_paths.append(os.path.join(root, file))

for root, dirs, files in os.walk(path_annot):
    for file in files:
        if file.endswith(".mat"):  
            annot_paths.append(os.path.join(root, file))

image_paths.sort()
annot_paths.sort()

image_size = 224 

images = []
targets = []

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

for i in range(len(annot_paths)):
    annot = scipy.io.loadmat(annot_paths[i])["box_coord"][0]
    top_left_x, top_left_y = annot[2], annot[0]
    bottom_right_x, bottom_right_y = annot[3], annot[1]

    image = Image.open(image_paths[i])
    (w, h) = image.size[:2]

    image = transform(image)

    images.append(image)

    target = torch.tensor([
        float(top_left_x) / w,
        float(top_left_y) / h,
        float(bottom_right_x) / w,
        float(bottom_right_y) / h,
    ])
    targets.append(target)

x_train = torch.stack(images[:int(len(images) * 0.8)])
y_train = torch.stack(targets[:int(len(targets) * 0.8)])

x_test = torch.stack(images[int(len(images) * 0.2):])
y_test = torch.stack(targets[int(len(targets) * 0.8):])

class projection_layer(nn.Module):
    def __init__(self,projection_dim:int,hidden_unit:int,dropout:float) -> None:
        super().__init__()
        self.hidden_units=hidden_unit
        self.projection_dim=projection_dim
        self.Drop_out=nn.Dropout(dropout)
        self.linear1=nn.Linear(projection_dim,hidden_unit[0])
        self.linear2=nn.Linear(hidden_unit[0],hidden_unit[1])

    def forward(self,x):
        out1=self.linear1(x)
        out2=self.Drop_out(out1)
        out3=self.linear2(out2)
        out4=self.Drop_out(out3)
        return out4

	
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchEncoder(nn.Module):
    def __init__(self, patch_size:int,input_size:int,projection_dim:int,num_patches:int):
        super().__init__()
        self.input_size=input_size[1]
        self.projection_layer=nn.Linear((3*patch_size*patch_size),projection_dim)
        self.postion_embedding=nn.Embedding(num_patches,projection_dim)
        self.patch_size=patch_size
        self.num_patches=num_patches

    def forward(self, images):
        positions = torch.unsqueeze(torch.arange(0, self.num_patches, 1), dim=0)
        positions = positions.to(device)
        batch_size, channels, height, width = images.size()
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size,num_patches_h * num_patches_w,self.patch_size *self.patch_size * channels)
        patches = patches.permute(0,1,2)
        out=self.projection_layer(patches)
        encoded=out+self.postion_embedding(positions)
        return encoded
        

    def extra_repr(self):
        return 'patch_size={}'.format(self.patch_size)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))  
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(std + self.eps) + self.bias
    
class residualconnections(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,x,sublayer):
        x=x+(sublayer(x))
        return x
    
class MultiHeadAttenionBlock(nn.Module):
    def __init__(self,projection_dim:int,dropout:float,num_heads:int):
        super().__init__()
        self.num_heads=num_heads
        self.projection_dim=projection_dim


        self.d_k=projection_dim//num_heads
        self.w_q=nn.Linear(projection_dim,projection_dim)
        self.w_k=nn.Linear(projection_dim,projection_dim)
        self.w_v=nn.Linear(projection_dim,projection_dim)

        self.w_o=nn.Linear(projection_dim,projection_dim)
        self.dropout=nn.Dropout(dropout)


    def attenion(key,query,value,dropout:nn.Dropout):
        d_k=query.shape[-1]
        attenion_score=(query@key.transpose(-2,-1))/math.sqrt(d_k) 
        attenion_score=attenion_score.softmax(dim=-1)     
        attenion_score=dropout(attenion_score)    
        return  (attenion_score@value),attenion_score
         

    def forward(self,q,k,v):
        query=self.w_k(q)
        key=self.w_q(k)
        value=self.w_v(v)
        

        query=query.view(query.shape[0],query.shape[1],self.num_heads,self.d_k).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.num_heads,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.num_heads,self.d_k).transpose(1,2)
        


        x,self.attenion_score=MultiHeadAttenionBlock.attenion(key,query,value,self.dropout)

        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.num_heads*self.d_k)

        return self.w_o(x)

class  TransformerBlock(nn.Module):

    def __init__(self,self_attention_block:MultiHeadAttenionBlock,dropout:float,hidden_units:int,projection_dim:int,layer_normalization:LayerNormalization,projection_layer1=projection_layer):
        super().__init__()
        self.layer_normalization=LayerNormalization()
        self.self_attention_block=self_attention_block
        self.residual_connections=nn.ModuleList([residualconnections() for _ in range(2)])
        self.drop_out=dropout
        self.hidden_units=hidden_units
        self.projection_dim=projection_dim
        self.projection_layer=projection_layer1

    def forward(self,x):
        x=self.layer_normalization(x)
        x=self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x))
        x=self.layer_normalization(x)    
        x=self.residual_connections[1](x,lambda x:self.projection_layer(x))
        return x
    
class mlp(nn.Module):
    def __init__(self,projection_dim:int,mlp_units:int,drop_out:float,num_p:int):
        super().__init__()
        self.projection_dim=projection_dim
        self.num_p=num_p
        self.mlp_unit=mlp_units
        self.Drop_out=nn.Dropout(drop_out)
        self.linear1=nn.Linear(self.projection_dim*self.num_p,mlp_units[0])
        self.linear2=nn.Linear(mlp_units[0],mlp_units[1])
        self.linear3=nn.Linear(mlp_units[1],mlp_units[2])
        self.linear4=nn.Linear(mlp_units[2],mlp_units[3])
        self.linear5=nn.Linear(mlp_units[3],mlp_units[4])
        self.linear6=nn.Linear(mlp_units[4],4)
   
    def forward(self,x):
        out1=self.linear1(x)
        out2=self.Drop_out(out1)
        out3=self.linear2(out2)
        out4=self.Drop_out(out3)
        out5=self.linear3(out4)
        out6=self.Drop_out(out5)
        out7=self.linear4(out6)
        out8=self.Drop_out(out7)
        out9=self.linear5(out8)
        out10=self.Drop_out(out9)
        out11=self.linear6(out10)
        return out11
        
class Bounding_box(nn.Module):
    def __init__(self,drop_out:float,layer_normalization:LayerNormalization,mlp1=mlp) -> None:
        super().__init__()
        self.drop_out_layer=nn.Dropout(drop_out)
        self.layer_normalization=LayerNormalization()
        self.mlp1=mlp1
    def forward(self,x):
        x=self.layer_normalization(x)
        x=self.drop_out_layer(torch.flatten(x,start_dim=1))
        x=self.mlp1(x)
        return x
    
class VITdetector(nn.Module):
    def __init__(self,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return (x)    
    
def create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
):
    vit=[]
    encoded_patches=PatchEncoder(patch_size,input_shape,projection_dim,num_patches)
    vit.append(encoded_patches)
    for i in range(transformer_layers):
        self_attention=MultiHeadAttenionBlock(projection_dim,0.1,num_heads)
        p_layer=projection_layer(projection_dim,transformer_units,0.3)
        Transformer_block=TransformerBlock(self_attention,0.1,transformer_units,projection_dim,LayerNormalization(),p_layer)
        vit.append(Transformer_block)
    mlp1=mlp(projection_dim,mlp_head_units,0.3,num_patches)
    bounding_box=Bounding_box(0.3,LayerNormalization(),mlp1)
    vit.append(bounding_box)


    vit_module=nn.ModuleList(vit)

    return vit_module

	
image_size=224
patch_size=32
input_shape = (image_size, image_size, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4

# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]

transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]

model_layer_list = create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
)

model=VITdetector(model_layer_list)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  #
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")