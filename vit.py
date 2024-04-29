import torch
import torch.nn as nn
import torchvision.transforms as transforms
from data import myDataset
from torch.utils.data import DataLoader
import math

#---CREATE PATCHES---
class patchify(nn.Module):
    def __init__(self,patch_size,projection_dim,num_patches):
        super().__init__()
        self.projection_layer = nn.Linear((3*patch_size*patch_size),projection_dim)
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self,images):
        positions = torch.unsqueeze(torch.arange(0,self.num_patches,1),dim=0)
        batch,channels,h,w = images.size()
        h_patches = h//self.patch_size
        w_patches = w//self.patch_size
        patches = images.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size)
        patches = patches.reshape(batch,h_patches*w_patches,self.patch_size*self.patch_size*channels)
        out = self.projection_layer(patches)
        return out
    
#---MULTI-HEAD ATTENTION---
class multiAttention(nn.Module):
    def __init__(self,emb_size,heads):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.d_head = emb_size//heads
        self.k = nn.Linear(self.d_head,self.d_head) #->key matrix
        self.q = nn.Linear(self.d_head,self.d_head) #->query matrix
        self.v = nn.Linear(self.d_head,self.d_head) #->value matrix
        self.w_o = nn.Linear(emb_size,emb_size)     #->reconstruction matrix

    def forward(self,query,key,value,mask):
        batch = query.shape[0] #-->get batch dimension
        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]  #-->all 3 are usually same
        #splitting the embeddings
        query = query.reshape(batch,query_len,self.heads,self.d_head)
        key = key.reshape(batch,key_len,self.heads,self.d_head)
        value = value.reshape(batch,value_len,self.heads,self.d_head)
        #passing them through k,q,v matrices
        query = self.q(query) 
        key = self.k(key)
        value = self.v(value)
        #taking dot product of query and key
        energy = torch.einsum("nqhd,nkhd->nhqk",[query,key]) #->output:(batch,heads,query_len,key_len)
        if mask is not None:
            energy = energy.masked_fill(mask==0,float("-1e15"))
        att = torch.softmax(energy/math.sqrt(self.emb_size),dim=1)
        out = torch.einsum("nhqk,nlhd->nqhd",[att,value]).reshape(batch,query_len,self.emb_size) #->output:(batch,query_len,emb_size)
        out = self.w_o(out)
        return out

class Transformer_block(nn.Module):
    def __init__(self,emb_size,heads,dropout,fwd_exp):
        super().__init__()
        self.self_att = multiAttention(emb_size,heads)
        self.norm1 = nn.LayerNorm(emb_size) #->after multi head attention
        self.norm2 = nn.LayerNorm(emb_size) #->after feed forward network
        self.ffn = nn.Sequential(nn.Linear(emb_size,fwd_exp*emb_size),
                                 nn.ReLU(),
                                 nn.Linear(fwd_exp*emb_size,emb_size)) #->in original paper fwd_exp is 4 times 
        #i.e., they had emb_size of 512 and ffn dimensions of 4*512 = 2048
        self.dropout = nn.Dropout(dropout)

    def forward(self,query,key,value,mask):
        att_out = self.self_att.forward(query,key,value,mask) 
        skip1_out = self.dropout(self.norm1(att_out+query))   #->1st skip connection
        ffn_out = self.ffn(skip1_out)  
        skip2_out = self.dropout(self.norm2(ffn_out+skip1_out)) #->2nd skip connection
        return skip2_out
    
class Encoder(nn.Module):
    def __init__(self,emb_size,blocks,heads,device,fwd_exp,dropout,seqLen):
        super().__init__()
        self.emb_size = emb_size
        self.posEmbeddings = nn.Embedding(seqLen,emb_size)   #->positional embeddings
        self.layers = nn.ModuleList( [Transformer_block(emb_size,heads,dropout,fwd_exp) for _ in range(blocks)] ) #->stacks of encoder blocks
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self,x,mask):
        batch,seqLen,_ = x.shape
        positions = torch.arange(0,seqLen).expand(batch,seqLen)
        x = self.dropout(x+self.posEmbeddings(positions))
        for layer in self.layers:
            x = layer(x,x,x,mask)
        return x

class vit(nn.Module):
    def __init__(self,patch_size , num_blocks , num_heads , device , fwd_exp , dropout , num_patches , projection_dim):
        super().__init__()
        self.encoder = Encoder(projection_dim , num_blocks , num_heads , device , fwd_exp , dropout , num_patches)
        self.patch_convertor = patchify(patch_size , projection_dim , num_patches)
        """self.fcn1 = nn.Linear(patch_size,patch_size/2)
        self.fcn2 = nn.Linear(patch_size/2,patch_size/4)
        self.fcn3 = nn.Linear(patch_size/4,patch_size/8)
        self.fcn4 = nn.Linear(patch_size/8,patch_size/16)
        self.fcn5 = nn.Linear(patch_size/16,patch_size/32)"""
        #self.test = nn.Linear(128,4)
        self.fcn1 = nn.Linear(128,64)
        self.fcn2 = nn.Linear(64,32)
        self.fcn3 = nn.Linear(32,16)
        self.fcn4 = nn.Linear(16,8)
        self.fcn5 = nn.Linear(8,4)

    def forward(self,img):
        patches = self.patch_convertor(img)
        encOut  = self.encoder(patches , None)
        x = encOut
        out1 = self.fcn1(x)
        out2 = self.fcn2(out1)
        out3 = self.fcn3(out2)
        out4 = self.fcn4(out3)
        out5 = self.fcn5(out4)
        #x = self.test(x)
        return out5

#---HYPER-PARAMETERS
img_size = 224
lr = 0.0001
batch_size = 20
patch_size = 14
num_patches = (img_size//patch_size)**2
num_blocks = 3
num_heads = 4
fwd_exp = 4
device = "cpu"
dropout = 0.3
projection_dim = 128

"""model = vit(patch_size,num_blocks,num_heads,device,fwd_exp,dropout,num_patches,projection_dim)
print(model(torch.randn((1,3,224,224)).shape))"""

"""

#---GET DATA---
imgs_path = "/vision _transformer/caltech-1011/101_ObjectCategories/101_ObjectCategories/airplanes"
annots_path = "/vision _transformer/caltech-1011/Annotations/Annotations/Airplanes_Side_2"
img_size = 256
data = myDataset(imgs_path,annots_path,img_size)

#---DATALOADERS---
trainLoader = DataLoader(data , batch_size=20 , shuffle=True)

#---TRAINING LOOP---
"""

if __name__=="__main__":
    inp = torch.randn((1,3,224,224))
    enc = patchify(14,48,(224//7)**2)
    model = vit(patch_size=14,num_blocks=3,num_heads=4,device="cpu",fwd_exp=4,dropout=0.2,num_patches=(224//7)**2,projection_dim=128)
    print(model(torch.randn((1,3,224,224))).shape)
