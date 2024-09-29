import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.init as init
import math


class lookupPreNorm(nn.Module):
    def __init__(self, dim, inner_dim):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim)
        self.fn = nn.Linear(dim,inner_dim,bias=False)
        init.xavier_normal_(self.fn.weight,gain=math.sqrt(5.0)) 
    def forward(self, x):
        return self.norm(self.fn(x))
    
class MHBC_look_com(nn.Module):              
    def __init__(self, dim,id_num, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.id_num=id_num
        self.scale = dim_head ** -0.5
        self.dim=dim
        self.attend = nn.Softmax(dim=-1)
        self.dropout=dropout
        self.to_q = lookupPreNorm(dim,inner_dim)
        self.to_k = lookupPreNorm(dim,inner_dim)
        self.to_v = nn.Linear(dim,inner_dim,bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()


 

    def forward(self, x,y): #x是大的lookup token,作为key和value；y是小的compress token，作为query
        b, n, _, h = *y.shape, self.heads
        win_num=n//self.id_num #一个窗口的动态信息的id维是16，n/16表示有多少个窗口
        q = self.to_q(y)      
        k = self.to_k(x)        
        v=self.to_v(x)

        q = rearrange(q, 'b (w n) (h d) -> (b w) h n d',w=win_num, h=h)   #多头注意力
        k=  rearrange(k, 'b (w n) (h d) -> (b w) h n d', w=win_num,h=h)   
        v=  rearrange(v, 'b (w n) (h d) -> (b w) h n d', w=win_num,h=h) 



        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b w) h n d -> b (w n) (h d)',w=win_num,h=h)
        y=y+self.to_out(out) #小token获得大token的信息
        return y,attn
    
class MHBC_com_look(nn.Module):              
    def __init__(self, dim,id_num, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.id_num=id_num
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_v = lookupPreNorm(dim,inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()


    def forward(self, x,y,attn): #x是大的lookup token；y是小的compress token，作为value;attn是MHBC_look_com计算出的attention矩阵
        b, n, _, h = *y.shape, self.heads

        win_num=n//self.id_num


        v=self.to_v(y)

        v=  rearrange(v, 'b (w n) (h d) -> (b w) h n d', w=win_num,h=h)     


  
        out = einsum('b h j i, b h j d -> b h i d', attn, v)

        out = rearrange(out, '(b w) h n d -> b (w n) (h d)',w=win_num)
        x=x+self.to_out(out) #大token获得小token的信息
        return x
    

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):              
    def __init__(self, dim, id_num,heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.id_num=id_num
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        init.xavier_uniform_(self.to_qkv.weight)  


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()
 

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        win_num=n//self.id_num
        qkv = self.to_qkv(x).chunk(3, dim=-1)           
        q, k, v = map(lambda t: rearrange(t, 'b (w n) (h d) -> (b w) h n d', w=win_num,h=h), qkv)          # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b w) h n d -> b (w n) (h d)',w=win_num)
        return self.to_out(out)+x
        

class vitblock(nn.Module):
    def __init__(self, dim, heads,id_num, dim_head, hidden_dim, dropout=0.): #hidden_dim为两层mlp的中间隐层神经元个数,对于小token，设定中间层翻4倍，对于大token，设定中间层为一半
        super().__init__()
        self.transformer=nn.Sequential(
            PreNorm(dim, Attention(dim, heads=heads,id_num=id_num, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, hidden_dim, dropout=dropout))
        )
                
    
    def forward(self, x):
        x=x+self.transformer(x) 
        '''
        要写x=x+self.transformer(x)而不是x+=self.transformer(x)，不然会报错，血的教训
        '''
        return x
    


class lookupvitblock(nn.Module):              
    def __init__(self, dim,id_num, heads=8, dim_head=64,dropout=0.1):
        super().__init__()

        self.dim=dim

        self.smalldim=4*dim #小token的mlp中间维度
        self.bigdim=dim//2 #大token的mlp中间维度

        self.heads = heads
        self.id_num=id_num

        self.dim_head=dim_head

        self.dropout=dropout

        self.look_to_com=MHBC_look_com(self.dim,self.id_num, self.heads, self.dim_head,self.dropout)

        self.vitblock=vitblock(self.dim, self.id_num,self.heads, self.dim_head, self.smalldim, dropout)

        self.com_to_look=MHBC_com_look(self.dim,self.id_num, self.heads, self.dim_head, self.dropout)

        self.out=PreNorm(self.dim,FeedForward(self.dim,self.bigdim, self.dropout))

    def forward(self,x,y): #x为大的lookup token；y为小的compress token
        y,attn=self.look_to_com(x,y)
        y=self.vitblock(y)
        x=self.com_to_look(x,y,attn)
        x=self.out(x)
        return x,y
    

#判断t是否是元组，如果是，直接返回t；如果不是，则将t复制为元组(t, t)再返回。用来处理当给出的图像尺寸或块尺寸是int类型（如224）时，直接返回为同值元组（如(224, 224)）
def pair(t):
    return t if isinstance(t, tuple) else (t, t)





    
class vediolookupvit(nn.Module):
    def __init__(self,image_size, patch_size, depth,channels=3,M_token=16,dim=768,  heads=8, dim_head=128, dropout=0.1): #M_token即为compresstoken的数量，采样于大token
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert  image_height % patch_height ==0 and image_width % patch_width == 0
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 2*channels * patch_height * patch_width
        self.channels=channels
        self.cnn1=nn.Sequential(
            nn.Conv2d(self.channels,2*self.channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(2*self.channels,2*self.channels,1,1,0),
        )
        
        self.cnn2=nn.Sequential(
            nn.Conv2d(self.channels,2*self.channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(2*self.channels,2*self.channels,1,1,0),
        )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )
        self.M_token=M_token
        self.depth=depth
        self.dim=dim
        self.heads=heads
        self.dim_head=dim_head
        self.dropout=dropout
        self.drop=nn.Dropout(self.dropout)
        self.pos_embedding_pic= nn.Parameter(torch.randn(1, num_patches, dim)) #位置编码
        #self.pos_embedding_vedio = nn.Parameter(torch.randn(1, 16, dim)) #中间维度
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(lookupvitblock(self.dim, self.M_token,self.heads, self.dim_head, self.dropout))
    
    def forward(self, img,vedio): #直接前向
        b,f, c, a,b = img.shape
        x=None
        for i in range(f):
            if f%2==0:
                img1=self.to_patch_embedding(self.cnn1(img[:,i]))
            else:
                img1=self.to_patch_embedding(self.cnn2(img[:,i]))
            b,n,_=img1.shape
            img1+= self.drop(self.pos_embedding_pic[:, :n]) 
            if x is None:
                x=img1
            else:
                x=torch.cat([x,img1],dim=1)


        y=vedio
        for block in self.layers:
            x1,y1 = block(x,y) 
            x=x1+x
            y=y1+y
        x=rearrange(x, 'b (w n) d -> b w n d',w=f//2)
        y=rearrange(y, 'b (w n) d -> b w n d',w=f//2)
        return x,y #输出为(batch,wim_num,token_num,emdedding)
    


# class vediolookupvit(nn.Module):
#     def __init__(self,image_size, patch_size, depth,channels=3,M_token=10,dim=768,  heads=8, dim_head=128, dropout=0.1): #M_token即为compresstoken的数量，采样于大token
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)
#         assert  image_height % patch_height ==0 and image_width % patch_width == 0
#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = 2*channels * patch_height * patch_width
#         self.channels=channels
#         self.cnn1=nn.Sequential(
#             nn.Conv2d(self.channels,2*self.channels,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(2*self.channels,2*self.channels,1,1,0),
#         )
        
#         self.cnn2=nn.Sequential(
#             nn.Conv2d(self.channels,2*self.channels,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(2*self.channels,2*self.channels,1,1,0),
#         )

#         self.cnn3=nn.Sequential(
#             nn.Conv2d(self.channels,2*self.channels,3,1,1),
#             torch.nn.BatchNorm2d(2*self.channels),
#             nn.ReLU(),
#             nn.Conv2d(2*self.channels,4*self.channels,8,2,6),
#             torch.nn.BatchNorm2d(4*self.channels),
#             nn.ReLU(),
#             nn.Conv2d(4*self.channels,8*self.channels,9,2,6),
#             torch.nn.BatchNorm2d(8*self.channels),
#             nn.ReLU(),
#             nn.Conv2d(8*self.channels,16*self.channels,1,1,0),
#             torch.nn.BatchNorm2d(16*self.channels),
#             nn.ReLU(),
#             nn.Conv2d(16*self.channels,16*self.channels,3,1,1),
#         )
#         self.cnn4=nn.Conv2d(self.channels,16*self.channels,20,4,16)
#         self.to_patch_embedding_vedio =nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=15, p2=15),
#             nn.LayerNorm(16*self.channels*15*15)
#         ) 
#         self.out1=nn.Sequential( nn.Linear(16*self.channels*15*15, 4096),
#             nn.Tanh(),
#             nn.Linear(4096,1024),
#             nn.Tanh(),
#             nn.Linear(1024,dim)
#         )
#         self.out2=nn.Linear(16*self.channels*15*15,dim)
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
#             nn.Linear(patch_dim, dim)
#         )
#         self.M_token=M_token
#         self.depth=depth
#         self.dim=dim
#         self.heads=heads
#         self.dim_head=dim_head
#         self.dropout=dropout
#         self.drop=nn.Dropout(self.dropout)
#         self.add_token=nn.Parameter(torch.randn(1,4, dim))
#         self.pos_embedding_pic= nn.Parameter(torch.randn(1, num_patches, dim)) #位置编码
#         self.pos_embedding_vedio= nn.Parameter(torch.randn(1, num_patches, dim)) #位置编码
#         self.layers = nn.ModuleList([])
#         for _ in range(self.depth):
#             self.layers.append(lookupvitblock(self.dim, self.M_token,self.heads, self.dim_head, self.dropout))
    
#     def forward(self, img,vedio): #直接前向
#         b,f, c, a,b = img.shape
#         b,v_f, c, a,b = vedio.shape
#         x=None
#         y=None
#         for i in range(f):
#             if f%2==0:
#                 img1=self.to_patch_embedding(self.cnn1(img[:,i]))
#             else:
#                 img1=self.to_patch_embedding(self.cnn2(img[:,i]))
#             b,n,_=img1.shape
#             img1+= self.drop(self.pos_embedding_pic[:, :n]) 
#             if x is None:
#                 x=img1
#             else:
#                 x=torch.cat([x,img1],dim=1)
  
#         for i in range(v_f):
#             v1=self.to_patch_embedding_vedio(self.cnn3(vedio[:,i])+self.cnn4(vedio[:,i]))
#             v1=self.out1(v1)+self.out2(v1)
#             b,n,_=v1.shape
#             v1+= self.drop(self.pos_embedding_vedio[:, :n]) 
#             v1=rearrange(v1, 'b (w n) d -> b w n d',n=self.M_token)
#             v1=v1.mean(dim=1)
#             if y is None:
#                 y=v1
#             else:
#                 y=torch.cat([y,v1],dim=1)

#         for block in self.layers:
#             x1,y1 = block(x,y) 
#             x=x1+x
#             y=y1+y
#         x=rearrange(x, 'b (w n) d -> b w n d',w=f//2)
#         y=rearrange(y, 'b (w n) d -> b w n d',w=f//2)

#         return x,y #输出为(batch,wim_num,token_num,emdedding)
    
# class lookupvit_rebuild_decoder(nn.Module):
#     def __init__(self,batchsize,channels,deep,embedding,dropout): #M_token即为compresstoken的数量，采样于大token
#         super().__init__()
#         self.batch=batchsize
#         self.channels=channels
#         self.deep=deep
#         self.embedding=embedding

#         self.id_up=nn.Sequential(
#             nn.Dropout(dropout),    
#             nn.Linear(self.embedding,2048),
#             nn.ReLU(),
#             nn.Linear(2048,4096),
#             nn.ReLU(),
#             nn.Linear(4096,12288), 
#         )
#         self.up1=nn.Linear(self.embedding,12288)