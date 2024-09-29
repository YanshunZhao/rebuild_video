import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from lookupvit import *
from cnn_rebuild import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
num_gpu=torch.cuda.device_count()  
ids=[i for i in range(num_gpu)]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

import myutils
import glob
import random
path2data="/mnt/store/zys/hmdb51"
sub_folder="video"
sub_folder_jpg="hmdb51_jpg"
path2ajpgs=os.path.join(path2data, sub_folder_jpg)

n_frames=20

#2. 每个视频的帧都存储在与该视频同名的文件夹中。调用myutils中的get_vids辅助函数来获得一个视频文件名和标签列表:
all_vids, all_labels, catgs=myutils.get_vids(path2ajpgs)

print(len(all_vids)) #打印所有视频数，所有便签数和所有种类数

all_vids, all_labels, catgs=myutils.get_vids(path2ajpgs)
print(len(all_vids), len(all_labels), len(catgs))

all_pic=[]
all_vedio=[]
all_label=[]
flag=0
for i in range(len(all_vids)):
    path2imgs=glob.glob(all_vids[i]+"/*.jpg")
    path2imgs.sort()
    last_elements = path2imgs[12:]  
    path2imgs=path2imgs[:2]+last_elements+path2imgs[2:12]
    if len(path2imgs)>=n_frames:
        all_vedio.append(path2imgs[1:4]+path2imgs[6:9]+path2imgs[11:14]+path2imgs[16:19])
        all_pic.append([path2imgs[i]for i in [1,4,5, 9, 10,14,15,19]] )
        all_label.append(all_labels[i]) #根据当前索引获取对应的标签,并使用 labels_dict 这个字典进行转换。（文本转数字）


print(len(all_vedio))
print(len(all_pic))
print(len(all_label))



#3. 定义一个Python字典来保存标签的数值
labels_dict={}
ind=0
for uc in catgs:
	labels_dict[uc]=ind
	ind+=1
print(labels_dict)

num_classes=51
all_pic_copy=[p for p, v,l in zip(all_pic,all_vedio,all_label) if labels_dict[l]<num_classes]
all_vedio_copy=[v for p, v,l in zip(all_pic,all_vedio,all_label) if labels_dict[l]<num_classes]
all_labels_copy=[l for p, v,l in zip(all_pic,all_vedio,all_label) if labels_dict[l]<num_classes]

all_pic=all_pic_copy
all_vedio=all_vedio_copy
all_label=all_labels_copy


#4. 将数据集分为训练数据集和测试数据集
from sklearn.model_selection import StratifiedShuffleSplit
sss=StratifiedShuffleSplit(n_splits=2,test_size=0.1,random_state=10)#random_state表示随机数种子
train_indx,test_indx=next(sss.split(all_vedio,all_label))
import sys  
def print_long_list(label, lst):  
    print(f"{label}: [{', '.join(map(str, lst))}]")  
# 将标准输出重定向到文件  
orig_stdout = sys.stdout  
f = open('/mnt/store/zys/rebuild_video/lookupvitrankpooling/result_pic/data_split.txt', 'w',buffering=1)  
sys.stdout = f  

# 在这里编写你的 Python 代码  
print_long_list('train:',train_indx)
for _ in range(5):
    print()
print_long_list('test:',test_indx)

# 恢复标准输出  
sys.stdout = orig_stdout  
f.close()  

train_pic=[all_pic[ind] for ind in train_indx]
train_ids=[all_vedio[ind] for ind in train_indx]
train_labels=[all_label[ind] for ind in train_indx]


test_pic=[all_pic[ind] for ind in test_indx]
test_ids=[all_vedio[ind] for ind in test_indx]
test_labels=[all_label[ind] for ind in test_indx]


from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torch
import numpy as np
import random


#2. 定义数据集类
class VideoDataset(Dataset):
	def __init__(self,pic,ids,labels,transform1,transform2):
		self.transform_video=transform1 #处理中间几帧
		self.transform_pic=transform2 #处理首位两帧
		self.pic=pic
		self.ids=ids
		self.labels=labels
	def __len__(self):
		return len(self.ids)
	def __getitem__(self,idx):
		label=labels_dict[self.labels[idx]]
		path2imgs=self.ids[idx]

		

		frames=[]
		for p2i in path2imgs:
			frame=Image.open(p2i)
			frames.append(frame) # Image.open() 函数读取每个图像,并将其添加到 frames 列表中。

		seed=np.random.randint(1e9)
		frames_tr=[]
		for frame in frames:
			random.seed(seed)
			np.random.seed(seed)
			frame=self.transform_video(frame)
			frames_tr.append(frame)
		if len(frames_tr)>0:
			frames_tr=torch.stack(frames_tr)


		path2pic=self.pic[idx]
		pics=[]
		for p2i in path2pic:
			pi=Image.open(p2i)
			pics.append(pi) # Image.open() 函数读取每个图像,并将其添加到 frames 列表中。

		seed=np.random.randint(1e9)
		pic_tr=[]
		for pi in pics:
			random.seed(seed)
			np.random.seed(seed)
			pi=self.transform_pic(pi)
			pic_tr.append(pi)
		if len(pic_tr)>0:
			pic_tr=torch.stack(pic_tr)
		return pic_tr,frames_tr, label




h_video,w_video=112,112
h_pic,w_pic=224,224
mean=[0,0,0]
std=[1,1,1] #定义了图像RGB三通道的均值和标准差


# 定义噪声函数  
def add_gaussian_noise(image, mean=0, std=0.2):  
    noise = torch.randn_like(image) * std + mean  
    return image + noise  

# 定义颜色扰动函数  
#分别表示亮度，对比度，饱和度，色调
def color_jitter_bright(image, brightness=0.2): 
    color_jitter_bright = transforms.ColorJitter(brightness=brightness,   
                                          )  
    return color_jitter_bright(image)  

def color_jitter_contract(image, contrast=0.2): 
    color_jitter_contract = transforms.ColorJitter( 
                                          contrast=contrast,  
                                          )  
    return color_jitter_contract(image)  

def color_jitter_saturation(image, saturation=0.2): 
    color_jitter_saturation = transforms.ColorJitter(
                                          saturation=saturation,   
                                          )  
    return color_jitter_saturation(image)  

def color_jitter_hue(image, hue=0.1): 
    color_jitter_hue = transforms.ColorJitter(
                                          hue=hue)  
    return color_jitter_hue(image)  
		
#4. 为训练集定义变换函数
import torchvision.transforms as transforms
train_vedio_transformer=transforms.Compose([
					transforms.Resize((h_video,w_video)), #输入图像resize到112x112像素
					#transforms.RandomHorizontalFlip(p=0.5), #以 50% 的概率对输入图像进行水平翻转
					#transforms.RandomAffine(degrees=0, translate=(0.1,0.1)), #输入图像进行随机仿射变换,平移范围为 ±10%
					transforms.ToTensor(),
					])

trian_pic_transformer=transforms.Compose([
				transforms.Resize((h_pic,w_pic)),
				#transforms.RandomHorizontalFlip(p=0.5), #以 50% 的概率对输入图像进行水平翻转
				#transforms.RandomAffine(degrees=0, translate=(0.1,0.1)), #输入图像进行随机仿射变换,平移范围为 ±10%
				transforms.ToTensor(),
				#transforms.RandomApply([add_gaussian_noise], p=0.5),  
                transforms.RandomApply([color_jitter_bright], p=0.5), 
				transforms.RandomApply([color_jitter_contract], p=0.5), 
				transforms.RandomApply([color_jitter_saturation], p=0.5), 
				transforms.RandomApply([color_jitter_hue], p=0.5), 
				])		

test_vedio_transformer=transforms.Compose([
					transforms.Resize((h_video,w_video)), #输入图像resize到112x112像素
					transforms.ToTensor(),
					])

test_pic_transformer=transforms.Compose([
				transforms.Resize((h_pic,w_pic)),
				transforms.ToTensor(),
				])	

#5. 实例化数据集类
train_ds=VideoDataset(train_pic,train_ids,train_labels,train_vedio_transformer,trian_pic_transformer)
print(len(train_ds))

#6. 获取train_ds中的一个数据
pic,vedio,label=train_ds[1]
if len(vedio)>0:
	print(pic.shape)
	print(vedio.shape)






#9. 实例化数据集类test_ds
test_ds=VideoDataset(test_pic,test_ids,test_labels,test_vedio_transformer,test_pic_transformer)
print(len(test_ds))

#10.  获取test_ds中的一个数据
pic,vedio,label=test_ds[1]
if len(vedio)>0:
	print(pic.shape)


# 定义collate_fn_3dcnn辅助函数
def collate_fn(batch):#这个函数主要用于3D卷积神经网络的数据加载过程中。它接收一个batch参数,这个参数是一个由样本组成的列表,每个样本包含图像和标签
	pic_batch,imgs_batch,label_batch=list(zip(*batch)) #将batch中的图像和标签分别取出,并转换为列表形式。
	pic_batch=[pic for pic in pic_batch if len(pic)>0]
	imgs_batch=[imgs for imgs in imgs_batch if len(imgs)>0]

	label_batch=[torch.tensor(l) for l,imgs in zip(label_batch,imgs_batch) if len(imgs)>0]

	pic_tensor=torch.stack(pic_batch)
	imgs_tensor=torch.stack(imgs_batch)
	imgs_tensor=torch.transpose(imgs_tensor,2,1) #图像列表转换为一个4D的张量并转置，最终得到(batch_size, channels, depth, height, width)

	labels_tensor=torch.stack(label_batch) 
	return pic_tensor,imgs_tensor,labels_tensor # 返回值是处理后的图像张量和标签张量


	
	
#1. 定义数据加载器
batch_size=256

train_dl=DataLoader(train_ds, batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle=False,collate_fn=collate_fn)

def weighted_sum_with_cosine_similarity(a, other):  
    # 计算余弦相似度  
    weight=[]
    for i in range(len(other)):
        weight.append(torch.cosine_similarity(a, other[i], dim=1).unsqueeze(1))


    weight=torch.cat(weight,dim=1)

    similarity_weights = torch.softmax(weight, dim=1)  



    weighted_sum=None
    for i in range(len(other)):
        if weighted_sum is None:
            weighted_sum =similarity_weights[:,i].unsqueeze(1) * other[i]
        else:
            weighted_sum+=similarity_weights[:,i].unsqueeze(1) * other[i]


    return weighted_sum 

from cnn_rebuild import *

video_ende=cnn_rebuild(3)
now_video=video_ende.state_dict()  
state_dict = torch.load('/mnt/store/zys/rebuild_video/lookupvitrankpooling/cnn_rebuild_pretrain.pth')  
new_state_dict = {}  
for k, v in state_dict.items():  
    if k.startswith('module.'):  
        name = k[7:] # 删除 'module.' 前缀  
        new_state_dict[name] = v  
    else:  
        new_state_dict[k] = v  

video_ende.load_state_dict(new_state_dict)  
video_ende = nn.DataParallel(video_ende, device_ids=ids) 


video_encoder=video_ende.module.encoder
video_decoder=video_ende.module.decoder
video_encoder = nn.DataParallel(video_encoder, device_ids=ids) 
video_decoder = nn.DataParallel(video_decoder, device_ids=ids) 
video_encoder=video_encoder.to(device)
video_decoder=video_decoder.to(device)
for param in video_ende.parameters():  
    param.requires_grad = False  


id_ende=idrebuild(16,512,10,14,14,10,768)
now_video=id_ende.state_dict()  
state_dict = torch.load('/mnt/store/zys/rebuild_video/lookupvitrankpooling/cnn_id_pretrain.pth')  
new_state_dict = {}  
for k, v in state_dict.items():  
    if k.startswith('module.'):  
        name = k[7:] # 删除 'module.' 前缀  
        new_state_dict[name] = v  
    else:  
        new_state_dict[k] = v  

id_ende.load_state_dict(new_state_dict)  
id_ende = nn.DataParallel(id_ende, device_ids=ids) 


id_encoder=id_ende.module.encoder
id_decoder=id_ende.module.decoder
id_encoder = nn.DataParallel(id_encoder, device_ids=ids) 
id_decoder = nn.DataParallel(id_decoder, device_ids=ids) 
id_encoder=id_encoder.to(device)
id_decoder=id_decoder.to(device)
for param in id_ende.parameters():  
    param.requires_grad = False  
   
  

from lookupvit import *
class lookvit_rebuild(nn.Module):
    def __init__(self,image_size, patch_size, depth,channels=3,M_token=16,dim=768,  heads=8, dim_head=128, dropout=0.1): #M_token即为compresstoken的数量，采样于大token
        super().__init__()
        self.encoder=vediolookupvit(image_size, patch_size, depth,channels=3,M_token=16,dim=768,  heads=8, dim_head=128, dropout=0.1)
        self.decoder=iddecoder(16,512,10,14,14,10,768)
    def forward(self,x,y):
        x,y=self.encoder(x,y)
        y=rearrange(y, 'b w n d -> (b w) n d')
        out_latent=self.decoder(y)
        return out_latent
vit=lookvit_rebuild(224, 16, depth=12)


now_video=vit.state_dict()  
state_dict = torch.load('/mnt/store/zys/rebuild_video/lookupvitrankpooling/lookupvit_pretrain.pth')  
new_state_dict = {}  
for k, v in state_dict.items():  
    if k.startswith('module.'):  
        name = k[7:] # 删除 'module.' 前缀  
        new_state_dict[name] = v  
    else:  
        new_state_dict[k] = v  

vit.load_state_dict(new_state_dict)  
vit = nn.DataParallel(vit, device_ids=ids) 
vit=vit.to(device)


# for param in vit.parameters():  
#     param.requires_grad = False  

vit_encoder=vit.module.encoder

vit_encoder = nn.DataParallel(vit_encoder, device_ids=ids) 
vit_encoder=vit_encoder.to(device)


for param in vit_encoder.parameters():  
    param.requires_grad = False  

from vit_class import*
class classifier(nn.Module):
    def __init__(self,num_class,dim=768,  dropout=0.3):
        super().__init__()
        self.model_out=ViT(num_class, dim, 2, 8,2*dim,  channels=3, dim_head=64, x_dropout=dropout, emb_dropout=0.2)

    def forward(self,x,y):
        #out=torch.cat([x,y],dim=2)
        out=y
        # b,win_num,t,e=out.shape
        # out=out.view(b,win_num,t//24,24,e).mean(dim=2)
        out = rearrange(out, 'b w h d -> b (w h) d')
        # win=[]
        # for i in range(win_num):
        #     win.append(out[:,i])
        # cls_tokens = repeat(self.cls_token, '() d -> b d', b=b)
        # out=weighted_sum_with_cosine_similarity(cls_tokens,win)
        # out=self.fla(out)
        # out=self.out(out)+self.down(out)

        return self.model_out(out)

model=classifier(51)
# state_dict = torch.load('vitclass_vedio.pth')  
# new_state_dict = {}  
# for k, v in state_dict.items():  
#     if k.startswith('module.'):  
#         name = k[7:] # 删除 'module.' 前缀  
#         new_state_dict[name] = v  
#     else:  
#         new_state_dict[k] = v  
# model.load_state_dict(new_state_dict)  

model = nn.DataParallel(model, device_ids=ids) 
model=model.to(device)


# 1. 定义损失函数、优化器和学习率计划:
import torch.optim as optim  
from torch.optim.lr_scheduler import ExponentialLR  
loss_func=nn.CrossEntropyLoss(reduction="mean")
lr=1e-4

opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0005)

lr_scheduler = ExponentialLR(opt, gamma=0.92) 

#2. 调用myutils中的train_val辅助函数训练模型
params_train={
	"num_epochs":80,
	"loss_func":loss_func,
	"train_dl":train_dl,
}


from tqdm import trange
epochs=params_train['num_epochs']
pbar = trange(epochs,smoothing=0)
loss_func=params_train['loss_func']
loss_hist=[]
acc_hist=[]
acc=0
for epoch in pbar:
    l_h=[]

    for pictures,vedio, labels in train_dl:
        pictures=pictures.to(device)
        vedio=vedio.to(device)

        labels=labels.to(device)
        vedio=torch.cat([id_encoder(video_encoder(vedio[:,:,0:3])),id_encoder(video_encoder(vedio[:,:,3:6])),id_encoder(video_encoder(vedio[:,:,6:9])),id_encoder(video_encoder(vedio[:,:,9:]))],dim=1)
        opt.zero_grad()
        x,y=vit_encoder(pictures,vedio)
        out=model(x,y)
        loss=loss_func(out,labels)
        if torch.isnan(loss):  
            print(f"Loss is NaN at epoch {epoch}. Training stopped.")  
        else:
            loss.backward()
            opt.step()  
            l_h.append(loss.item())


        for param_group in opt.param_groups:
            lr=param_group["lr"]

        pbar.set_postfix(**{'loss': loss.item(), 'lr': lr,'accuracy':acc})

    with torch.no_grad():
        true_data,all_data=0,0
        for pictures,vedio, labels in test_dl:
            pictures=pictures.to(device)
            vedio=vedio.to(device)

            labels=labels.to(device)
            vedio=torch.cat([id_encoder(video_encoder(vedio[:,:,0:3])),id_encoder(video_encoder(vedio[:,:,3:6])),id_encoder(video_encoder(vedio[:,:,6:9])),id_encoder(video_encoder(vedio[:,:,9:]))],dim=1)

            x,y=vit_encoder(pictures,vedio)
            _,out=model(x,y).max(1)
            true_data+= (out == labels).sum().item()  
            all_data+=len(labels)
        acc=true_data/all_data
        if (epoch+1)%5==0:
            print(acc)
        acc_hist.append(acc)
        torch.save(model, 'vitclass_vedio.pth')
    if (epoch+1)%2==0:
        lr_scheduler.step()


    
    loss_hist.append(np.mean(np.array(l_h)))


plt.plot([i for i in range(len(loss_hist))],loss_hist)
plt.savefig('/mnt/store/zys/rebuild_video/lookupvitrankpooling/result_pic/loss')
plt.show()

plt.plot([i for i in range(len(acc_hist))],acc_hist)
plt.savefig('/mnt/store/zys/rebuild_video/lookupvitrankpooling/result_pic/acc')
plt.show()

orig_stdout = sys.stdout  
f = open('/mnt/store/zys/rebuild_video/lookupvitrankpooling/result_pic/result.txt', 'w',buffering=1)  
sys.stdout = f  

# 在这里编写你的 Python 代码
print_long_list('loss:',loss_hist) 
for _ in range(5):
    print() 
print_long_list('acc:',acc_hist)

# 恢复标准输出  
sys.stdout = orig_stdout  
f.close()  

vit_encoder.eval()

model.eval()  

true_data,all_data=0,0
flag=0
for pictures,vedio, labels in train_dl:
    pictures=pictures.to(device)
    vedio=vedio.to(device)

    labels=labels.to(device)
    vedio=torch.cat([id_encoder(video_encoder(vedio[:,:,0:3])),id_encoder(video_encoder(vedio[:,:,3:6])),id_encoder(video_encoder(vedio[:,:,6:9])),id_encoder(video_encoder(vedio[:,:,9:]))],dim=1)

    x,y=vit_encoder(pictures,vedio)
    _,out=model(x,y).max(1)
    true_data+= (out == labels).sum().item()  
    all_data+=len(labels)
print(true_data/all_data)

true_data,all_data=0,0
flag=0
y_true_list = []  
y_pred_list = [] 
for pictures,vedio, labels in test_dl:
    pictures=pictures.to(device)
    vedio=vedio.to(device)

    labels=labels.to(device)
    vedio=torch.cat([id_encoder(video_encoder(vedio[:,:,0:3])),id_encoder(video_encoder(vedio[:,:,3:6])),id_encoder(video_encoder(vedio[:,:,6:9])),id_encoder(video_encoder(vedio[:,:,9:]))],dim=1)

    x,y=vit_encoder(pictures,vedio)
    _,out=model(x,y).max(1)
    true_data+= (out == labels).sum().item()  
    all_data+=len(labels)
    y_true_list.extend(labels.tolist())  
    y_pred_list.extend(out.tolist())  
print(true_data/all_data)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_list, y_pred_list)  
print("混淆矩阵:")  
print(cm)  

# 可视化混淆矩阵  
plt.figure(figsize=(12, 10))  
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  
plt.title('confusion matrix')  
plt.colorbar()  
tick_marks = np.arange(len(set(y_true_list)))  
plt.xticks(tick_marks, [str(i) for i in set(y_true_list)], rotation=90)  
plt.yticks(tick_marks, [str(i) for i in set(y_true_list)])  

fmt = 'd'  
thresh = cm.max() / 2.  
for i in range(cm.shape[0]):  
    for j in range(cm.shape[1]):  
        plt.text(j, i, format(cm[i, j], fmt),  
                 horizontalalignment="center",  
                 color="white" if cm[i, j] > thresh else "black")  

plt.ylabel('true')  
plt.xlabel('pre')  
plt.tight_layout() 
plt.savefig('/mnt/store/zys/rebuild_video/lookupvitrankpooling/result_pic/confusion matrix') 
plt.show()  

for i in range(51):
    if cm[i][i]==0:
        print(i)