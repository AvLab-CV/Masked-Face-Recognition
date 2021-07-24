import face_model
import cv2
import numpy as np
import os
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv

#######################################################################################################################
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--ga-model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
#parser.add_argument('--MaskPath', default=R'../Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_crop_pose',help='')
parser.add_argument('--MaskPath', default=R'E:/database/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_crop',help='')
#parser.add_argument('--WoMaskPath', default=R'../Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop_choose',help='')
parser.add_argument('--WoMaskPath', default=R'E:/database/self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop',help='')
parser.add_argument('--SavePath', default='./feature/',help='path to save feature txt file.')
parser.add_argument('--FeaturePath', default='',help='path to load feature txt file.')
parser.add_argument('--FeaturePath2', default='',help='path to load feature txt file.')
#parser.add_argument('--Protocal', default='./Protocol_verification/AFDB_0829_all.csv',help='')   
parser.add_argument('--Protocal', default='./RMFRD/Protocol_verification/AFDB_final_Jeff.csv',help='') 
parser.add_argument('--Method', default=['Arcface','Cosface','Sphereface','Arcface_mask_mouth_ori','Cosface_mask_mouth_ori','Sphereface_mask_mouth_ori'],help='Model to test.')
args = parser.parse_args()
#######################################################################################################################

def feature(args,model,file,path):
    try:
        if path == 'Mask':
            img = cv2.imread('{}/{}'.format(args.MaskPath,file))
            Savepath = '{}/Mask'.format(args.SavePath)
            name = file.split('/')[0] + '_' + file.split('/')[-1].split('.')[0]
        elif path == 'WoMask':
            img = cv2.imread('{}/{}'.format(args.WoMaskPath,file))
            Savepath = '{}/WoMask'.format(args.SavePath)
            name = file.split('/')[0] + '_' + file.split('/')[-1].split('.')[0]
        if not os.path.exists(Savepath): os.makedirs(Savepath)
        if os.path.isfile('{}/{}.txt'.format(Savepath,name)):return
        img_2 = np.transpose(img, (2,0,1))
    ##detect face
    # img_New,img_New2,points_new,bbox_new = model.get_input(img)
    # img = img_New[:,:,:,0].reshape(3,112,112)
    except ValueError:
        if path == 'Mask':
            img = cv2.imdecode(np.fromfile('{}/{}'.format(args.MaskPath,file), dtype=np.uint8), -1)
        elif path == 'WoMask':
            img = cv2.imdecode(np.fromfile('{}/{}'.format(args.WoMaskPath,file), dtype=np.uint8), -1)
        img_2 = np.transpose(img, (2,0,1))
    
    f = model.get_feature(img_2)
    ##Save feature
    with open ('{}/{}.txt'.format(Savepath,name),'w') as fff:
        for j in f:
            fff.writelines(str(j))
            fff.writelines('\n')
    return

def main():
    for method in args.Method:
        ##################Load Model######################
        args.model = './model/{}/model,0'.format(method)
        args.ga_model = args.model
        args.FeaturePath = './feature/AFDB/{}_AFDB/Mask'.format(method)
        args.FeaturePath2 = './feature/AFDB/{}_AFDB/Womask'.format(method)
        args.SavePath = './feature/AFDB/{}_AFDB'.format(method)
        if not os.path.exists(args.SavePath): os.makedirs(args.SavePath)
        model = face_model.FaceModel(args)
        ##################################################

        ##################Save feature####################
        # with open('./Protocol_identification/AFDB_gallery.csv','r',newline='') as f:
        #     rows = csv.reader(f)
        #     for row in rows:
        #         feature(args,model,row[0],'Mask')
        # with open('./Protocol_identification/AFDB_probe.csv','r',newline='') as f:
        #     rows = csv.reader(f)
        #     for row in rows:
        #         feature(args,model,row[0],'WoMask')
        with open(args.Protocal,'r',newline='') as f:
            rows = csv.reader(f)
            next(rows)
            for row in rows:
                feature(args,model,row[0],'Mask')
                feature(args,model,row[1],'WoMask')
        print('Save {} features'.format(method))
        ##################################################


if __name__ == "__main__":
    main()