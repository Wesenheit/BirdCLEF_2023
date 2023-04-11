import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import PureAudio,ProcessPipe,score_predictions,CosineWarmupScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
from transformer import ViTBackbone,BirdAttention
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
torch.set_float32_matmul_precision("high")
@torch.no_grad()
def eval(model,data,N,device,preprocess):
    model.eval()
    np.random.seed(42)
    idx=np.random.randint(0,len(data),size=(N,))
    Y=[]
    Y_pred=[]
    for id in idx:
        (x,y) = data[id]
        x = x.to(device)
        x = preprocess(x.reshape(1,-1),aug=False)
        x = x.unsqueeze(1)
        y_pred = model(x)
        y = F.one_hot(y.reshape(1,-1),data.n_class).float()
        Y.append(y.squeeze(0).cpu())
        Y_pred.append(y_pred.cpu())
    Y = torch.stack(Y).squeeze(1)
    Y_pred = torch.stack(Y_pred).squeeze(1)
    return score_predictions(Y_pred,Y,pad=0)


audio=PureAudio("train_metadata.csv","../data/")
loader=DataLoader(audio,128,True,num_workers=3)
device="cuda"
size=384
dimen=512
preprocessor=ProcessPipe(32000,15340,size).to(device)
backbone=ViTBackbone(dimen,3*dimen,8,5,8**2,12**2).to(device)
model=BirdAttention(audio.n_class,dimen,backbone).to(device)
optim=torch.optim.Adam(model.parameters(),3e-5)
N=20
N_per_epo=7
shed=CosineWarmupScheduler(optim,N//10,N)
model=torch.compile(model,backend="inductor")
for n in range(1,N+1):
    loss_ep=0
    model.train() 
    for _ in range(N_per_epo):
        for X,Y in tqdm(loader):
            optim.zero_grad()
            X = preprocessor(X.to(device))
            X = X.unsqueeze(1)
            pred = model(X)
            Y = F.one_hot(Y,audio.n_class).float().to(device)
            loss=F.binary_cross_entropy_with_logits(pred,Y,reduction="sum")
            loss.backward()
            optim.step()
            loss_ep+=loss.item()/len(audio)
    shed.step()
    (score,acc)=eval(model,audio,500,device,preprocessor)
    print("epoch: {:.0f}, loss: {:.4f}, score: {:.4f}, acc: {:.4f}".format(n,loss_ep/N_per_epo,score,acc))

torch.save(model.state_dict(),"test.tc")