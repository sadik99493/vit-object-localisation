import torch
import torch.nn as nn
import math

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
    def __init__(self,inp_vocab_size,emb_size,blocks,heads,device,fwd_exp,dropout,seqLen):
        super().__init__()
        self.inp_vocab_size = inp_vocab_size
        self.emb_size = emb_size
        self.wordEmbeddings = nn.Embedding(inp_vocab_size,emb_size)  #->input embeddings
        self.posEmbeddings = nn.Embedding(seqLen,emb_size)   #->positional embeddings
        self.layers = nn.ModuleList( [Transformer_block(emb_size,heads,dropout,fwd_exp) for _ in range(blocks)] ) #->stacks of encoder blocks
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self,x,mask):
        batch,seqLen = x.shape
        positions = torch.arange(0,seqLen).expand(batch,seqLen).to(self.device)
        x = self.dropout(self.wordEmbeddings(x)+self.posEmbeddings(positions))
        for layer in self.layers:
            x = layer(x,x,x,mask)
        return x
    
class decoderBlock(nn.Module):
    def __init__(self,emb_size,heads,fwd_exp,device,dropout):
        super().__init__()
        self.att = multiAttention(emb_size,heads)
        self.norm = nn.LayerNorm(emb_size)
        self.transformerBlock = Transformer_block(emb_size,heads,dropout,fwd_exp)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,value,key,inpMask,outMask): #-->value and key are nothing but encoder output and outMask is the causal mask for decoder
        att_out = self.att(x,x,x,outMask)
        skip1_out = self.dropout(self.norm(att_out+x))  #->1st skip connection
        finalOut = self.transformerBlock(skip1_out,key,value,inpMask) #->contains the remaining skip connections and ffns etc
        return finalOut
    
class Decoder(nn.Module):
    def __init__(self,out_vocab_size,emb_size,blocks,heads,fwd_exp,dropout,device,seqLen):
        super().__init__()
        self.wordEmbeddings = nn.Embedding(out_vocab_size,emb_size)
        self.posEmbeddings = nn.Embedding(seqLen,emb_size)
        self.device = device
        self.layers = nn.ModuleList( [decoderBlock(emb_size,heads,fwd_exp,device,dropout) for _ in range(blocks)])
        self.fcn = nn.Linear(emb_size,out_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,encOut,inpMask,outMask):
        batch,seqLen = x.shape
        positions = torch.arange(0,seqLen).expand(batch,seqLen)
        x = self.dropout(self.wordEmbeddings(x)+self.posEmbeddings(positions))
        for layer in self.layers:
            x = layer(x,encOut,encOut,inpMask,outMask)
        out = self.fcn(x)
        return torch.softmax(out,dim=-1)

class build(nn.Module):
    def __init__(self,inp_vocab_size,out_vocab_size,inp_pad_index,out_pad_index,emb_size=512,blocks=6,heads=8,fwd_exp=4,dropout=0.1,device="cpu",seqLen=150):
        super().__init__()
        self.encoder = Encoder(inp_vocab_size,emb_size,blocks,heads,device,fwd_exp,dropout,seqLen)
        self.decoder = Decoder(out_vocab_size,emb_size,blocks,heads,fwd_exp,dropout,device,seqLen)
        self.inp_pad_index = inp_pad_index  #->index of padding token in input tokenizer
        self.out_pad_index = out_pad_index  #->index of padding token in output tokenizer
        self.device = device

    def get_inp_mask(self,inp):
        mask = (inp!=self.inp_pad_index).unsqueeze(1).unsqueeze(2)  #->mask for encoder
        return mask.to(self.device)
    
    def get_out_mask(self,out):
        batch,outLen = out.shape
        mask = torch.tril(torch.ones((outLen,outLen))).expand(batch,1,outLen,outLen)  #->causal mask for decoder
        return mask.to(self.device)
    
    def forward(self,inp,out):
        inpMask = self.get_inp_mask(inp)
        outMask = self.get_out_mask(out)
        encOut = self.encoder(inp,inpMask)
        out = self.decoder(out,encOut,inpMask,outMask)
        return out

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = torch.tensor([[1,3,2,5,3,0,0],[9,2,5,1,0,0,0]]).to(device)   #-->input sequence length = 7
    out = torch.tensor([[4,1,2,4,4,3,5],[1,2,2,7,7,3,5]]).to(device)   #-->output sequence length = 7
    inp_pad=0    #index of padding token in source vocabulary
    out_pad=0    #index of padding token in target vocabulary
    inp_size=10  #-->input vocabulary length
    out_size=10  #-->output vocabulary length
    model = build(inp_size,out_size,inp_pad,out_pad).to(device)
    final = model(inp,out[:,:-1])    #--> ( output seq length x output vocabulary size ) --> (7 x 10)
    print(final)



