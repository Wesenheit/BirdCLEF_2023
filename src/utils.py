import torch 
import torchaudio
from torch.utils.data import DataLoader,Dataset
from typing import Optional
import pandas as pd
from torch import nn  as nn
from sklearn.metrics import average_precision_score,accuracy_score
import  numpy as np


class PureAudio(Dataset):
    def __init__(self,metadata,data_dir,length=5) -> None:
        super().__init__()
        self.data_dir=data_dir
        self.metadata=pd.read_csv(data_dir+metadata)
        self.n_class=len(self.metadata["primary_label"].unique())
        self.mapping={key:i for i,key in enumerate(self.metadata["primary_label"].unique())}
        self.sample=32000
        self.len=length

    def __getitem__(self, index) -> torch.tensor:
        bird_class=torch.tensor(self.mapping[self.metadata["primary_label"][index]])
        tensor,_=torchaudio.load(self.data_dir+"train_audio/"+self.metadata["filename"][index])
        tensor=tensor.squeeze(0)
        length=self.len*self.sample
        if tensor.shape[0]>length:
            index=torch.randint(tensor.shape[0]-length,size=())
            tensor=tensor[index:index+length]
        else:
            new_tensor=tensor
            while len(new_tensor)<length:
                new_tensor=torch.cat((new_tensor,tensor))
            tensor=new_tensor[:length]
        return tensor,bird_class

    def __len__(self):
        return len(self.metadata)
    
class ProcessPipe(nn.Module):
    def __init__(self,original,new_sample,size=128,freq=400) -> None:
        super().__init__()
        self.resample = torchaudio.transforms.Resample(orig_freq=original, new_freq=new_sample)
        self.spec_aug = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=80),
            torchaudio.transforms.TimeMasking(time_mask_param=80),
        )

        self.spectr = torchaudio.transforms.MelSpectrogram(n_mels=size,n_fft=freq)

    def forward(self,x,aug=True):
        x = self.resample(x)
        x = self.spectr(x)
        if aug:
            x = self.spec_aug(x)
        return x
    
def score_predictions(Y_pred,Y_true,pad=5):
    new_rows = []
    for i in range(pad):
        new_rows.append([1 for i in range(Y_pred.shape[1])])
    new_rows=np.array(new_rows)
    Y_pred=Y_pred.detach().cpu().numpy()
    Y_true=Y_true.detach().cpu().numpy()
    if pad>0:
        Y_true=np.concatenate((Y_true,new_rows),axis=0)
        Y_pred=np.concatenate((Y_pred,new_rows),axis=0)
    return (average_precision_score(Y_true,Y_pred,average='macro'),accuracy_score(np.argmax(Y_true,1),np.argmax(Y_pred,1)))


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor