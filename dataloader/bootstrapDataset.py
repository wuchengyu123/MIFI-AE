import os.path
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchio as tio

class CustomDataSet_OS(Dataset):
    def __init__(self,
                 csv_data,  # 临床csv特征文件
                 csv_data_col_names, # 使用特征列名
                 t_ct_dataset_path, # CT图像路径
                 is_test
                 ):
        super(CustomDataSet_OS, self).__init__()
        self.transform = tio.Compose([
            tio.RandomNoise(p=0.6),
            tio.RandomBiasField(p=0.3),
            tio.RandomFlip(),
            tio.RandomMotion(p=0.2),
            tio.OneOf({
                tio.RandomAffine():0.7,
                tio.RandomElasticDeformation(num_control_points=5):0.3
            },p=0.75)
        ])
        self.t_ct_dataset_path = t_ct_dataset_path
        self.csv_data_col_names = csv_data_col_names
        self.csv_data = csv_data
        self.patient_clinical_data = self.csv_data[[*csv_data_col_names]].values
        self.is_test = is_test

        self.csv_IBEX_CT_NAME = np.squeeze(self.csv_data.loc[:,['IBEX_CT_NAME']].values)


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        clinical_data = np.asarray(self.patient_clinical_data[index],dtype=np.float32)
        OSstatue = np.asarray(self.csv_data.loc[index,'OSstatue'],dtype=np.float32)
        OStime = np.asarray(self.csv_data.loc[index,'OStime'],dtype=np.float32)
        PatientID = np.asarray(self.csv_data.loc[index, 'PatientID'], dtype=np.int32)


        OStime = np.expand_dims(OStime,axis=-1)
        OSstatue = np.expand_dims(OSstatue,axis=-1)
        PatientID = np.expand_dims(PatientID, axis=-1)

        t_img = sitk.GetArrayFromImage(
                sitk.ReadImage(os.path.join(self.t_ct_dataset_path,
                               '{}_CT.nii.gz'.format(str(self.csv_IBEX_CT_NAME[index])))))

        t_img = np.expand_dims(t_img,axis=0)
        t_img = np.asarray(t_img,dtype=np.float32)


        return  {'t_img':t_img, 'OSstatue': OSstatue, 'clinical_data': clinical_data,'PatientID':PatientID, 'OStime': OStime}
