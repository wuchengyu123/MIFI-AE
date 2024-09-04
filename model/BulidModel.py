import torch
from torch import nn
from model.ResNet3D_BraTS import generate_model_BraTS_LF



class ResNet_Linformer(nn.Module):
    def __init__(self):
        super(ResNet_Linformer,self).__init__()

        self.resnet_linformer = generate_model_BraTS_LF(model_depth=50)

        self.fc = nn.Linear(448,1)
        self.fc_text = nn.Linear(216,224)
        #self.fc_img =

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)

    def forward(self, t_img, tabular_data):
        text_feat = tabular_data
        t_img_out, img_embed, text_embed = self.resnet_linformer(t_img,text_feat)
        text_embed = self.fc_text(text_embed)

        output = torch.concat((img_embed,text_embed),dim=1)
        risk_score = self.fc(output)
        return t_img_out[:,:,:42,...], img_embed, text_embed, risk_score

if __name__ == '__main__':
    x1 = torch.rand((3, 1, 128, 128, 128))

    info = torch.zeros((3, 54))
    resnet_transformer = ResNet_Linformer()
    output, _, text_embed,_ = resnet_transformer(x1, info)
    print(output.shape)
    print(text_embed.shape)
