import torch
import torch.nn as nn
from einops import rearrange

def conv3d_bn_relu(inch,outch,kernel_size,stride=1,padding=1): #卷积，激活函数为relu
    convlayer = torch.nn.Sequential(
        torch.nn.Conv3d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm3d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def conv3d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=1): #卷积，激活函数为sigmoid
    convlayer = torch.nn.Sequential(
        torch.nn.Conv3d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm3d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1): #转置卷积，激活函数为sigmoid
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm3d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_relu(inch,outch,kernel_size,stride=1,padding=1):#转置卷积，激活函数为relu
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm3d(outch),
        torch.nn.ReLU()
    )
    return convlayer


# class conRefineFireModelframe_14(torch.nn.Module):
#     def __init__(self, in_channels):
#         super(conRefineFireModelframe_14,self).__init__()
#         self.channels=in_channels
#         self.conv_stack0 = torch.nn.Sequential(
#             conv3d_bn_relu(self.channels,16,(3,12,12),(1,3,3),(1,10,10)),
#             conv3d_bn_relu(16,16,3)
#         )
#         self.conv_stack1 = torch.nn.Sequential(
#             conv3d_bn_relu(16,64,(1,7,7),(1,2,2),(0,5,5)),
#             conv3d_bn_relu(64,64,3)
#         )
#         self.conv_stack2 = torch.nn.Sequential(
#             conv3d_bn_relu(64,256,(2,5,5),(1,2,2),(1,4,4)),
#             conv3d_bn_relu(256,256,3)
#         )
#         self.conv_stack3 = torch.nn.Sequential(
#             conv3d_bn_relu(256,512,(1,1,1),(1,1,1),(0,0,0)),
#             conv3d_bn_relu(512,512,3)
#         )

#         self.frame1=deconv_relu(512,512,(5,1,1),(3,1,1),(2,0,0))

        
#         self.deconv_3 = deconv_relu(512,256,(3,3,3),(1,1,1),(1,1,1))
#         self.deconv_2 = deconv_relu(256+64,64,(3,5,5),(2,2,2), (1,4,4))
#         self.deconv_1 = deconv_relu(64+16,16,(6,7,7),(1,2,2), (2,5,5))
#         self.deconv_0 = deconv_relu(16+8,8,(6,12,12),(2,3,3), (3,10,10))

#         self.predict_3 = torch.nn.Conv3d(512,64,(3,1,1),(1,1,1),(1,0,0))
#         self.predict_2 = torch.nn.Conv3d(256+64,16,(4,4,4),(2,2,2),(1,1,1))
#         self.predict_1 = torch.nn.Conv3d(64+16,8,(5,5,5),(3,2,2),(2,2,2))

        
        
#         self.up_sample_3 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(64,64,(3,3,3),(1,1,1),(1,1,1)),
#             torch.nn.ReLU()
#         )
#         self.up_sample_2 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(16,16,(7,5,5),(5,4,4),(4,3,3)),
#             torch.nn.ReLU()
#         )
#         self.up_sample_1 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(8,8,(4,5,5),(3,4,4),(1,4,4),bias=False),
#             torch.nn.ReLU()
#         )

#         self.frame2=conv3d_bn_relu(8,self.channels,(12,1,1),(8,1,1),(3,0,0))

#         self.activation=nn.ReLU()

#         self.downres=torch.nn.Sequential(
#             torch.nn.Conv3d(self.channels,256,(3,12,12),(1,8,8),(1,2,2)),
#             nn.ReLU(),
#             nn.ConvTranspose3d(256,512,(6,1,1),(4,1,1),(2,0,0))
#         )
        
#         self.upres=torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(512,256,(1,12,12),(1,8,8),(0,2,2)),
#             nn.ReLU(),
#             nn.Conv3d(256,self.channels,(4,1,1),(2,1,1),(1,0,0))
#         )
        


#     def encoder(self, x):#下采样提取关键信息
#         conv1_out = self.conv_stack0(x)
#         conv2_out = self.conv_stack1(conv1_out)
#         conv3_out = self.conv_stack2(conv2_out)
#         conv4_out = self.conv_stack3(conv3_out)
#         frame1=self.frame1(conv4_out)

#         return frame1+self.downres(x)
    
#     def decoder(self, x): #从特征上采样到原始维度
#         deconv3_out = self.deconv_3(x)
#         predict_3_out = self.up_sample_3(self.predict_3(x))
#         concat3 = torch.cat([deconv3_out,predict_3_out],dim=1)

#         deconv2_out = self.deconv_2(concat3)
#         predict_2_out = self.up_sample_2(self.predict_2(concat3))
#         concat2 = torch.cat([deconv2_out,predict_2_out],dim=1)

#         deconv1_out = self.deconv_1(concat2)
#         predict_1_out = self.up_sample_1(self.predict_1(concat2))
#         concat1 = torch.cat([deconv1_out,predict_1_out],dim=1)


#         predict_out = self.deconv_0(concat1)
#         predict_out=self.frame2(predict_out)
#         return self.activation(predict_out+self.upres(x))
    

#     def forward(self,x):
#         latent = self.encoder(x) #编码得到特征
#         out = self.decoder(latent) #解码恢复原图像
#         return latent,out
    
class rebuild_encoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(rebuild_encoder,self).__init__()
        self.channels=in_channels
        self.conv_stack0 = torch.nn.Sequential(
            conv3d_bn_relu(self.channels,16,(3,12,12),(1,3,3),(1,10,10)),
            conv3d_bn_relu(16,16,3)
        )
        self.conv_stack1 = torch.nn.Sequential(
            conv3d_bn_relu(16,64,(1,7,7),(1,2,2),(0,5,5)),
            conv3d_bn_relu(64,64,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv3d_bn_relu(64,256,(2,5,5),(1,2,2),(1,4,4)),
            conv3d_bn_relu(256,256,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv3d_bn_relu(256,512,(1,1,1),(1,1,1),(0,0,0)),
            conv3d_bn_relu(512,512,3)
        )

        self.frame1=deconv_relu(512,512,(5,1,1),(3,1,1),(2,0,0))
        self.downres=torch.nn.Sequential(
            torch.nn.Conv3d(self.channels,256,(3,12,12),(1,8,8),(1,2,2)),
            nn.ReLU(),
            nn.ConvTranspose3d(256,512,(6,1,1),(4,1,1),(2,0,0))
        )
    def forward(self, x):#下采样提取关键信息
        conv1_out = self.conv_stack0(x)
        conv2_out = self.conv_stack1(conv1_out)
        conv3_out = self.conv_stack2(conv2_out)
        conv4_out = self.conv_stack3(conv3_out)
        frame1=self.frame1(conv4_out)

        return frame1+self.downres(x)
    
class rebuild_decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(rebuild_decoder,self).__init__()
        self.channels=in_channels
        self.deconv_3 = deconv_relu(512,256,(3,3,3),(1,1,1),(1,1,1))
        self.deconv_2 = deconv_relu(256+64,64,(3,5,5),(2,2,2), (1,4,4))
        self.deconv_1 = deconv_relu(64+16,16,(6,7,7),(1,2,2), (2,5,5))
        self.deconv_0 = deconv_relu(16+8,8,(6,12,12),(2,3,3), (3,10,10))

        self.predict_3 = torch.nn.Conv3d(512,64,(3,1,1),(1,1,1),(1,0,0))
        self.predict_2 = torch.nn.Conv3d(256+64,16,(4,4,4),(2,2,2),(1,1,1))
        self.predict_1 = torch.nn.Conv3d(64+16,8,(5,5,5),(3,2,2),(2,2,2))

        
        
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64,64,(3,3,3),(1,1,1),(1,1,1)),
            torch.nn.ReLU()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16,16,(7,5,5),(5,4,4),(4,3,3)),
            torch.nn.ReLU()
        )
        self.up_sample_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8,8,(4,5,5),(3,4,4),(1,4,4),bias=False),
            torch.nn.ReLU()
        )

        self.frame2=conv3d_bn_relu(8,self.channels,(12,1,1),(8,1,1),(3,0,0))

        self.activation=nn.ReLU()
        
        self.upres=torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512,256,(1,12,12),(1,8,8),(0,2,2)),
            nn.ReLU(),
            nn.Conv3d(256,self.channels,(4,1,1),(2,1,1),(1,0,0))
        )
    def forward(self, x): #从特征上采样到原始维度
        deconv3_out = self.deconv_3(x)
        predict_3_out = self.up_sample_3(self.predict_3(x))
        concat3 = torch.cat([deconv3_out,predict_3_out],dim=1)

        deconv2_out = self.deconv_2(concat3)
        predict_2_out = self.up_sample_2(self.predict_2(concat3))
        concat2 = torch.cat([deconv2_out,predict_2_out],dim=1)

        deconv1_out = self.deconv_1(concat2)
        predict_1_out = self.up_sample_1(self.predict_1(concat2))
        concat1 = torch.cat([deconv1_out,predict_1_out],dim=1)


        predict_out = self.deconv_0(concat1)
        predict_out=self.frame2(predict_out)
        return self.activation(predict_out+self.upres(x))
    

class cnn_rebuild(nn.Module):  
    def __init__(self,channels):  
        super(cnn_rebuild, self).__init__()  
        self.encoder = rebuild_encoder(channels)  
        self.decoder = rebuild_decoder(channels)
    def forward(self,x):
        latant=self.encoder(x)
        out=self.decoder(latant)
        return latant,out 

    
class idencoder(torch.nn.Module):
    def __init__(self, batchsize,channels,deep,h,w,id,embedding):
        super(idencoder,self).__init__()
        self.batch=batchsize
        self.channels=channels
        self.deep=deep
        self.h=h
        self.w=w
        self.id=id
        self.embedding=embedding

        self.conv_stack0 = torch.nn.Sequential(
            deconv_relu(512,128,4,2,3),
            deconv_relu(128,128,3)
        )
        self.conv_stack1 = torch.nn.Sequential(
            deconv_relu(128,64,5,1,(2,1,1)),
            deconv_relu(64,64,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            deconv_relu(64,32,(5,7,7),(1,1,1),(1,2,2)),
            deconv_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            deconv_relu(32,16,(1,1,1),(1,1,1),(0,0,0)),
            deconv_relu(16,16,3)
        )

        self.fla=nn.Sequential(
            nn.Flatten()  
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, 8*9*14*14))#位置编码，因为id大概为10,16维相当于离10最近且能被（16,18,28,28）整除


        self.id_out=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(8*9*14*14,4096), #相当于下采样了16倍，因为id大概为10,16维相当于离10最近且能被（16,18,28,28）整除
            nn.ReLU(),
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Linear(1024,self.embedding)
        )

        self.down1=deconv_relu(512,16,(4,6,6),2,2)
        self.down2=nn.Linear(8*9*14*14,self.embedding)

    def forward(self,x):
        conv1_out = self.conv_stack0(x)
        conv2_out = self.conv_stack1(conv1_out)
        conv3_out = self.conv_stack2(conv2_out)
        conv4_out = self.conv_stack3(conv3_out)+self.down1(x)
        fla=self.fla(conv4_out)
        fla=rearrange(fla, 'b (h d) -> b h d', h = 16)     #当于下采样了16倍，因为id大概为10,16维相当于离10最近且能被（16,18,28,28）整除  
        fla+=self.pos_embedding[:, :16]  
        id_embedding=self.id_out(fla)+self.down2(fla)
        return id_embedding
    
class iddecoder(torch.nn.Module):
    def __init__(self, batchsize,channels,deep,h,w,id,embedding):
        super(iddecoder,self).__init__()
        self.batch=batchsize
        self.channels=channels
        self.deep=deep
        self.h=h
        self.w=w
        self.id=id
        self.embedding=embedding

        self.id_up=nn.Sequential(
            nn.Dropout(0.2),    
            nn.Linear(self.embedding,1024),
            nn.ReLU(),
            nn.Linear(1024,4096),
            nn.ReLU(),
            nn.Linear(4096,8*9*14*14), 
        )


        self.deconv_3 = conv3d_bn_relu(16,64,(1,1,1),(1,1,1),(0,0,0))
        self.deconv_2 = conv3d_bn_relu(64+16,128,(6,8,8),2,3)
        self.deconv_1 = conv3d_bn_relu(128+32,384,3,1,1)


        self.predict_3 = nn.ConvTranspose3d(16,8,(3,1,1),(1,1,1),(1,0,0))
        self.predict_2 = nn.ConvTranspose3d(64+16,16,6,2,2)
        self.predict_1 = nn.ConvTranspose3d(128+32,64,(1,3,3),(1,1,1),(0,1,1))

        
        
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.Conv3d(8,16,(1,3,3),(1,1,1),(0,1,1)),
            torch.nn.ReLU()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.Conv3d(16,32,(6,8,8),4,(3,2,2)),
            torch.nn.ReLU()
        )
        self.up_sample_1 = torch.nn.Sequential(
            torch.nn.Conv3d(64,128,(3,1,1),(1,1,1),(1,0,0)),
            torch.nn.ReLU()
        )


        self.up1=nn.Linear(self.embedding,8*9*14*14)
        self.up2=conv3d_bn_relu(16,512,(4,8,8),2,(2,3,3))

    def forward(self,x):
        fea=self.id_up(x)+self.up1(x)
        fea=rearrange(fea, 'b h d -> b (h d)')
        fea=rearrange(fea, 'b (c d h w) -> b c d h w',c=16,d=18,h=28,w=28)

        deconv3_out = self.deconv_3(fea)
        predict_3_out = self.up_sample_3(self.predict_3(fea))
        concat3 = torch.cat([deconv3_out,predict_3_out],dim=1)

        deconv2_out = self.deconv_2(concat3)
        predict_2_out = self.up_sample_2(self.predict_2(concat3))
        concat2 = torch.cat([deconv2_out,predict_2_out],dim=1)

        deconv1_out = self.deconv_1(concat2)
        predict_1_out = self.up_sample_1(self.predict_1(concat2))
        concat1 = torch.cat([deconv1_out,predict_1_out],dim=1)


        return concat1+self.up2(fea)

class idrebuild(torch.nn.Module):
    def __init__(self, batchsize,channels,deep,h,w,id,embedding):
        super(idrebuild,self).__init__()
        self.batch=batchsize
        self.channels=channels
        self.deep=deep
        self.h=h
        self.w=w
        self.id=id
        self.embedding=embedding

        self.encoder=idencoder(self.batch,self.channels,self.deep,self.h,self.w,self.id,self.embedding)
        self.decoder=iddecoder(self.batch,self.channels,self.deep,self.h,self.w,self.id,self.embedding)
        
    
    def forward(self,x):
        latent = self.encoder(x) #编码得到特征
        out = self.decoder(latent) #解码恢复原图像
        return latent,out
