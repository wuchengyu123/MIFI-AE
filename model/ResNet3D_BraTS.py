import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .SwinUNETR import SwinBlock
from .CrossAttention import CrossAttention
import copy
from model import Linformer as LF


def get_inplanes():
    return [4,8,16,32]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


        
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, in_planes, planes, stride=1, downsample=None):
#         super().__init__()
#
#         self.conv1 = conv3x3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3x3(planes, planes)
#         self.bn2 = nn.BatchNorm3d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Linformer_Layer(nn.Module):
    """
    A wrapper function to accept LM tasks, inspired by https://github.com/lucidrains/sinkhorn-transformer
    """
    def __init__(self, num_tokens, input_size, channels,
                       dim_k=64, dim_ff=1024, dim_d=None,
                       dropout_ff=0.1, dropout_tokens=0.1, nhead=4, depth=1, ff_intermediate=None,
                       dropout=0.05, activation="gelu", checkpoint_level="C0",
                       parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False,
                       include_ff=True, w_o_intermediate_dim=None, emb_dim=None,
                       return_emb=False, decoder_mode=False, causal=False, method="learnable"):
        super(Linformer_Layer, self).__init__()
        emb_dim = channels if emb_dim is None else emb_dim

        self.input_size = input_size

        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = LF.PositionalEmbedding(emb_dim)
        self.linformer = LF.Linformer(input_size, channels, dim_k=dim_k,
                                   dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff,
                                   nhead=nhead, depth=depth, dropout=dropout, ff_intermediate=ff_intermediate,
                                   activation=activation, checkpoint_level=checkpoint_level, parameter_sharing=parameter_sharing,
                                   k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention, include_ff=include_ff,
                                   w_o_intermediate_dim=w_o_intermediate_dim, decoder_mode=decoder_mode, causal=causal, method=method)

        if emb_dim != channels:
            self.linformer = LF.ProjectInOut(self.linformer, emb_dim, channels)

        self.dropout_tokens = nn.Dropout(dropout_tokens)

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
        """
        # tensor = self.to_token_emb(tensor)
        tensor = self.pos_emb(tensor).type(tensor.type()) + tensor
        tensor = self.dropout_tokens(tensor)
        tensor = self.linformer(tensor, **kwargs)
        return tensor

class ResNet_Linformer(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 num_classes=256):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 1, 1),
                               padding=(6, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(2, 2, 2))
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(2, 2, 2))

        self.layer_up1 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(1, 1, 1))
        self.in_planes = 64
        self.layer_up2 = self._make_layer(block,
                                          block_inplanes[2],
                                          layers[2],
                                          shortcut_type,
                                          stride=(1, 1, 1))
        self.in_planes = 32
        self.layer_up3 = self._make_layer(block,
                                          block_inplanes[1],
                                          layers[1],
                                          shortcut_type,
                                          stride=(1, 1, 1))
        self.in_planes = 16
        self.layer_up4 = self._make_layer(block,
                                          block_inplanes[0],
                                          layers[0],
                                          shortcut_type,
                                          stride=(1, 1, 1))

        self.conv_16 = nn.Conv3d(in_channels=16,out_channels=64,kernel_size=8,stride=4,padding=2)
        self.conv_32 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2,padding=1)

        self.conv_1024to256 = nn.ConvTranspose3d(in_channels=64,out_channels=16,kernel_size=8,padding=2,stride=4)#nn.Conv3d(in_channels=64, out_channels=16, kernel_size=1, stride=1)
        self.conv_1024to512 = nn.ConvTranspose3d(in_channels=64,out_channels=32,kernel_size=4,padding=1,stride=2)#nn.Conv3d(in_channels=64, out_channels=32, kernel_size=1, stride=1)

        #self.swin_block1_enc = SwinBlock(dim=16, depth=1, num_heads=8, window_size=[8, 8, 8], drop_path=[1])
        #self.swin_block2_enc = SwinBlock(dim=32, depth=1, num_heads=8, window_size=[8, 8, 8], drop_path=[1])

        self.swin_block3_enc = SwinBlock(dim=64, depth=1, num_heads=8, window_size=[8, 8, 8], drop_path=[1])

        #self.swin_block1_dec = SwinBlock(dim=64, depth=1, num_heads=8, window_size=[8, 8, 8], drop_path=[1])
        #self.swin_block2_dec = SwinBlock(dim=32, depth=1, num_heads=8, window_size=[8, 8, 8], drop_path=[1])

        self.upsample1 = nn.ConvTranspose3d(in_channels=128,out_channels=64,kernel_size=4,padding=1,stride=2)
        self.upsample2 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.upsample3 = nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, padding=1, stride=2)
        self.upsample4 = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=4, padding=1, stride=2)

        self.CrossAtten1 = CrossAttention(heads=4,d_model=32,img_dim=16,txt_dim=54)
        self.CrossAtten2 = CrossAttention(heads=4,d_model=32,img_dim=32,txt_dim=54)
        self.CrossAtten3 = CrossAttention(heads=4, d_model=32, img_dim=64, txt_dim=54)

        self.conv_swin1 = nn.Sequential(
            #nn.Conv3d(16,1,kernel_size=1,stride=1),
            nn.AdaptiveAvgPool3d((1)),
            #nn.Flatten()
        )

        self.conv_swin2 = nn.Sequential(
            #nn.Conv3d(32, 1, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool3d((1)),
            #nn.Flatten()
        )

        self.conv_swin3 = nn.Sequential(
            #nn.Conv3d(64, 1, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool3d((1)),
            #nn.Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.lf1 = Linformer_Layer(num_tokens=1, input_size=3*6*16*16, channels=64)

    def linformer_layer(self, x1, x2, x3, lf):
        b1, c1, d1, h1, w1 = x1.size()
        b2, c2, d2, h2, w2 = x2.size()
        b3, c3, d3, h3, w3 = x3.size()

        x1 = x1.view(x1.size(0), x1.size(1), -1).transpose(1, 2) # 21504
        x2 = x2.view(x2.size(0), x2.size(1), -1).transpose(1, 2) # 5819
        x3 = x3.view(x3.size(0), x3.size(1), -1).transpose(1, 2) # 2312
        x = torch.cat((x1, x2, x3), dim=1)
        x = lf(x).transpose(1, 2)
        x1 = x[:, :, 0:d1 * h1 * w1].view(b1, c1, d1, h1, w1)
        x2 = x[:, :, d1 * h1 * w1:d1 * h1 * w1 + d2 * h2 * w2].view(b2, c2, d2, h2, w2)
        x3 = x[:, :, d1 * h1 * w1 + d2 * h2 * w2:d1 * h1 * w1 + d2 * h2 * w2 + d3 * h3 * w3].view(b3, c3, d3, h3, w3)
        return x1, x2, x3, x


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, text):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        if not self.no_max_pool:
            x1 = self.maxpool(x1)  # [3,64,32,32,32]
        #print('x1 shape:',x1.shape)


        x1 = self.layer1(x1) # [3,64,32,32,32]
        #print('x1 lf1:',x1.shape)

        x1_swin = x1#self.swin_block1_enc(x1)
        #print('x1 after swin1_enc:',x1_swin.shape)

        #print(self.conv_swin1(x1_swin).squeeze().shape)
        cross_x1, cross_txt1 = self.CrossAtten1(self.conv_swin1(x1_swin).squeeze(),text)

        x2 = self.layer2(x1_swin)
        #print('x1 lf2:', x2.shape)

        x2_swin = x2#self.swin_block2_enc(x2)
        #print('x1 after swin2_enc:', x2_swin.shape)

        cross_x2, cross_txt2 = self.CrossAtten2(self.conv_swin2(x2_swin).squeeze(),cross_txt1)


        x3 = self.layer3(x2_swin)
        #print('x1 lf3',x3.shape)

        x3_swin = self.swin_block3_enc(x3)
        #print('x1 after swin3_enc:', x3_swin.shape)

        cross_x3, cross_txt3 = self.CrossAtten3(self.conv_swin3(x3_swin).squeeze(), cross_txt2)


        x4 = self.layer4(x3_swin)
        #print('x1 lf4',x4.shape)


        #print('x1_lf 1112222333 shape',self.conv_16(x1_swin).shape)
        #print('x2_lf 1112222333 shape', self.conv_32(x2_swin).shape)
        x1_lf, x2_lf, x3_lf, _ = self.linformer_layer(self.conv_16(x1_swin), self.conv_32(x2_swin), x3_swin, self.lf1)

        # print('x3_lf shape', x3_lf.shape)



        xup1 = self.layer_up1(x4)
        #print('xup1',xup1.shape)

        x5 = self.upsample1(xup1)
        #print('x5 shape',x5.shape)
        x5_swin_dec = x5#self.swin_block1_dec(x5)
        x5_swin_dec = x5_swin_dec + x3_lf
        #print('x5_swin_dec',x5_swin_dec.shape)  #[1024,16,16,16]

######################################################
        xup2 = self.layer_up2(x5_swin_dec)
        #print('xup2',xup2.shape)

        x6 = self.upsample2(xup2)
        #print('x6 shape',x6.shape)

        x6_swin_dec = x6#self.swin_block2_dec(x6)
        #print('conv_1024to512(x2_lf).shape',self.conv_1024to512(x2_lf).shape)

        x6_swin_dec = x6_swin_dec + self.conv_1024to512(x2_lf)
        #print('x6_swin_dec',x6_swin_dec.shape)

###########################################################

        xup3 = self.layer_up3(x6_swin_dec)
        #print('xup3', xup3.shape)

        x7 = self.upsample3(xup3)
        #print('x7 shape', x7.shape)

        x7_swin_dec = x7
        #print('self.conv_1024to256(x1_lf).shape',self.conv_1024to256(x1_lf).shape)
        x7_swin_dec = x7_swin_dec + self.conv_1024to256(x1_lf)
        #print('x7_swin_dec', x7_swin_dec.shape)

###########################################################

        xup4 = self.layer_up4(x7_swin_dec)
        #print('xup4',xup4.shape)
        x_out = self.upsample4(xup4)
        #print('xout shape',x_out.shape)

        #img_out_embed = torch.concat((cross_x1,cross_x2,cross_x3),dim=1)
        #text_out_embed = torch.concat((cross_txt1,cross_txt2,cross_txt3),dim=1)

        outx1 = self.conv_swin1(x1_swin)
        outx1 = outx1.view(outx1.size(0),-1)

        outx2 = self.conv_swin2(x2_swin)
        outx2 = outx2.view(outx2.size(0),-1)

        outx3 = self.conv_swin3(x3_swin)
        outx3 = outx3.view(outx3.size(0),-1)
        # print(outx1.shape)
        # print(outx2.shape)
        # print(outx3.shape)


        img_out_embed = torch.concat((outx1,outx2,outx3,cross_x1,cross_x2,cross_x3),dim=1)
        text_out_embed = torch.concat((cross_txt1,cross_txt2,cross_txt3,text),dim=1)
        #print('img_out_embed shape',img_out_embed.shape)
        #print('text_out_embed shape',text_out_embed.shape)

        return x_out,img_out_embed,text_out_embed

def generate_model_BraTS_LF(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    model = ResNet_Linformer(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)

    return model

if __name__ == '__main__':
    x1 = torch.rand((3,1,42,128,128))

    info = torch.zeros((3,54))
    resnet_transformer = generate_model_BraTS_LF(model_depth=50)
    output,img_embed,text_embed = resnet_transformer(x1,info)
    print(output.shape)
    print(img_embed.shape)
    print(text_embed.shape)







