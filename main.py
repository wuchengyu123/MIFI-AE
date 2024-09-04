
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataloader.bootstrapDataset import CustomDataSet_OS
from NegitiveLogLikelihood import NegativeLogLikelihood
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from model.BulidModel import ResNet_Linformer
from utils import record_best_model
from metrics import c_index
import time
import numpy as np
print(torch.cuda.is_available())
print(torch.cuda.device_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')

#writer = SummaryWriter('logs')
csv_data_col_names  = ['gradient_glrlm_LongRunHighGrayLevelEmphasis_T',
     'gradient_ngtdm_Busyness_T',
     'lbp-2D_glrlm_LongRunHighGrayLevelEmphasis_T',
     'square_firstorder_StandardDeviation_T',
     'wavelet-LHL_glszm_LargeAreaHighGrayLevelEmphasis_T',
     'wavelet-HLH_firstorder_Kurtosis_T',
     'wavelet-HHL_firstorder_Median_T',
     'gradient_firstorder_10Percentile_N',
     'gradient_firstorder_Median_N',
     'wavelet-LLH_gldm_GrayLevelNonUniformity_N',
     'wavelet-LLH_glrlm_GrayLevelNonUniformity_N',
     'wavelet-LLH_glszm_LowGrayLevelZoneEmphasis_N',
     'wavelet-HLL_glszm_LargeAreaLowGrayLevelEmphasis_N',
     'wavelet-HLH_glcm_ClusterProminence_N',
     'wavelet-HHL_glrlm_LongRunHighGrayLevelEmphasis_N',
     'wavelet-HHH_glcm_Id_N',
     'wavelet-HHH_glszm_LargeAreaLowGrayLevelEmphasis_N',
     'wavelet-HHH_glszm_SizeZoneNonUniformityNormalized_N',
     'wavelet-HHH_glszm_SmallAreaEmphasis_N',
     '分化程度Group',
     '血小板体积分组',
     'PathTstage',
     'PathNstage',
     'PathMstage',
     'PathTNMstage',
     'Path8thstage',
     '肿瘤病理最长径',
     'CAO血红蛋白',
     'CAO血小板压积',
     'CAO总蛋白',
     'CAO肌酐',
     'CAO葡萄糖',
     'CAO钾',
     'CAO钠',
     'CAO钙',
     'CAO国际标准化比值',
     'CAO活化部分凝血活酶时间',
       'CAO红细胞压积',  #
       'CAO平均血小板体积',  #
       'CAO血小板分布宽度',
       'CAO中性粒细胞数目',  #
       'CAO总胆红素',  #
       'CAO直接胆红素',  #
       'KPS评分分级',
       'CAO平均RBC体积',
       'CAORBC分布宽度标准差',
       'CAO血小板数目',
       'CAO嗜碱性细胞数目',
       'CAO间接胆红素',
       'CAO白蛋白',
       'CAO球蛋白',
       'CAO丙氨酸氨基转移酶',
       'CAO尿素',
       'CAO纤维蛋白原',
                       ]

patient_id = 'PatientID'
event_times_col_name = 'OStime'
event_observed_col_name = 'OSstatue'

t_image_path = '/home/wuchengyu/another_GTV-T/all_set'

batch_size = 12
random_seeds  = [4,5,7,9,10]
Xy_data = pd.read_csv('Xy_all_hazard.csv',encoding='gbk')
columns = Xy_data.columns

X = Xy_data.drop(columns=['IBEX_CT_NAME','PatientID','OStime','OSstatue','PFStime','PFSstatue'])
y = Xy_data.loc[:,['IBEX_CT_NAME','PatientID','OStime','OSstatue','PFStime','PFSstatue']]

for seed in random_seeds:
    print(f'random seed:{seed}')
    print('-'*20)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
    train = pd.concat([y_train,X_train],axis=1)
    test = pd.concat([y_test,X_test],axis=1)

    train_data = pd.DataFrame(train.values,columns=train.columns)
    test_data = pd.DataFrame(test.values,columns=test.columns)

    train_set= CustomDataSet_OS(
            csv_data=train_data,
            csv_data_col_names=csv_data_col_names,
            t_ct_dataset_path= t_image_path,
            is_test=False
            )

    valid_set = CustomDataSet_OS(
        csv_data=test_data,
        csv_data_col_names=csv_data_col_names,
        t_ct_dataset_path=t_image_path,
        is_test=True
    )
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)



    lr = 1e-4 # 1e-5
    model= ResNet_Linformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20,eta_min=5e-6) # 8e-6
    criterion_regression_loss_with_censored = NegativeLogLikelihood(device=device,l2_reg=1.5)
    mseloss = nn.MSELoss()
    history_best_test_cindex = -1
    history_best_train_cindex = -1
    epochs = 10

    best_c_index = -1
    for epoch in range(epochs):

        final_pred_train = []
        OStime_train = []
        OSstatue_train = []
        batch_loss = 0
        batch_rec_loss = 0
        batch_div_loss = 0
        model.train()
        for batch in train_loader:
            t_img, OStime, OSstatue, clinical_data = batch['t_img'], batch[
                'OStime'], batch['OSstatue'], batch['clinical_data']
            #print(clinical_data_train.shape)
            t_img = t_img.to(device, dtype=torch.float32)

            #print(t_img.shape)
            OStime = OStime.to(device, dtype=torch.float32)
            OSstatue = OSstatue.to(device, dtype=torch.float32)
            clinical_data = clinical_data.to(device, dtype=torch.float32)

            t_img_out, img_embed, text_embed, risk_socre = model(t_img, clinical_data)

            surv_loss = criterion_regression_loss_with_censored(risk_socre, OStime, OSstatue, model)
            rec_loss = mseloss(t_img_out,t_img)
            div_loss = F.kl_div(img_embed.softmax(-1).log(), text_embed.softmax(-1), reduction='batchmean')

            loss = surv_loss + rec_loss + div_loss

            batch_loss += loss
            batch_rec_loss += rec_loss
            #batch_div_loss += div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_pred_train.append(-risk_socre.detach().cpu().numpy())
            OStime_train.append(OStime.detach().cpu().numpy())
            OSstatue_train.append(OSstatue.detach().cpu().numpy())
        final_pred_concat_train = np.concatenate(final_pred_train, axis=0)
        OStime_concat_train = np.concatenate(OStime_train, axis=0)
        OSstatue_concat_train = np.concatenate(OSstatue_train, axis=0)


        train_cindex = c_index(final_pred_concat_train, OStime_concat_train, OSstatue_concat_train)
        if train_cindex < 0.5:
            train_cindex = 1-train_cindex


        print('epoch{}/{} total_loss: {} div: {} rec_loss: {} train c-index: {} '.format(epoch,epochs,batch_loss/len(train_loader),batch_div_loss/len(train_loader),
                                                                                batch_rec_loss/len(train_loader),train_cindex),end=' ')


        final_pred_test = []
        OStime_test = []
        OSstatue_test = []
        model.eval()
        for batch in valid_loader:
            t_img,OStime, OSstatue, clinical_data =batch['t_img'],batch['OStime'], batch['OSstatue'], batch['clinical_data']
            OStime = OStime.to(device,dtype=torch.float32)
            OSstatue = OSstatue.to(device,dtype=torch.float32)
            clinical_data = clinical_data.to(device,dtype=torch.float32)


            t_img = t_img.to(device, dtype=torch.float32)

            with torch.no_grad():
                t_img_out, img_embed, text_embed,risk_socre = model(t_img,clinical_data)


                final_pred_test.append(-risk_socre.detach().cpu().numpy())
                OStime_test.append(OStime.detach().cpu().numpy())
                OSstatue_test.append(OSstatue.detach().cpu().numpy())

        final_pred_concat_test = np.concatenate(final_pred_test, axis=0)
        OStime_concat_test = np.concatenate(OStime_test, axis=0)
        OSstatue_concat_test = np.concatenate(OSstatue_test, axis=0)

        test_cindex = c_index(final_pred_concat_test,OStime_concat_test,OSstatue_concat_test)

        if test_cindex < 0.5:
            test_cindex = 1-test_cindex

        print('test c-index: {}'.format(test_cindex))

        if best_c_index < test_cindex and test_cindex > 0.65:
            best_c_index = test_cindex
            record_best_model(model, '{} seed{} best test c-index:{:.4f} train c-index:{:.4f}.pth'.format(time.strftime('%H-%M-%S', time.localtime(time.time())),
                                                                     seed,test_cindex,train_cindex), 'cross_valid file')
            print(f'best_test c-index in epoch{epoch}: {best_c_index}')

            if history_best_test_cindex < best_c_index:
                history_best_test_cindex = best_c_index
                history_best_train_cindex = train_cindex

    with open('best_cindex_OS_surv+rec.txt', 'a+') as f:
        f.write('seed'+str(seed))
        f.write(' best test cindex: '+str(history_best_test_cindex))
        f.write(' best train_cindex: '+str(history_best_train_cindex))
        f.write('\n')
