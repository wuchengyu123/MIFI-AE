import torch.nn as nn
import math
import torch
import torch.nn.functional as F


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, q_d_model, kv_d_model, dropout=0.5):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(q_d_model, d_model)
        self.v_linear = nn.Linear(kv_d_model, d_model)
        self.k_linear = nn.Linear(kv_d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, q_d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class CrossAttention(nn.Module):
    def __init__(self,heads,d_model,img_dim,txt_dim):
        super(CrossAttention, self).__init__()
        self.image_MHA = MultiHeadAttention(heads=heads, d_model=d_model, q_d_model=img_dim, kv_d_model=img_dim)
        self.text_MHA = MultiHeadAttention(heads=heads, d_model=d_model, q_d_model=txt_dim, kv_d_model=txt_dim)

        self.image_CrossMHA = MultiHeadAttention(heads=heads, d_model=d_model, q_d_model=img_dim, kv_d_model=txt_dim)
        self.text_CrossMHA = MultiHeadAttention(heads=heads, d_model=d_model, q_d_model=txt_dim, kv_d_model=img_dim)

    def forward(self,image_embed,text_embed):
        img_embed = self.image_MHA(image_embed,image_embed,image_embed)
        txt_embed = self.text_MHA(text_embed,text_embed,text_embed)

        img_output = self.image_CrossMHA(img_embed,txt_embed,txt_embed)
        text_output = self.text_CrossMHA(txt_embed,img_embed,img_embed)

        return img_output.squeeze(1),text_output.squeeze(1)

if __name__ == '__main__':
    x = torch.rand((2,54))
    y = torch.rand((2,128))
    model = CrossAttention(heads=4,d_model=128,img_dim=128,txt_dim=54)
    k = model(y,x)
    print(k[0].shape)