import torch


class LabelEncoder(torch.nn.Module):
    def __init__(self,**kwargs):
        super(LabelEncoder, self).__init__()


    def encode(self,x):
        return None, None, [None,None,x]
