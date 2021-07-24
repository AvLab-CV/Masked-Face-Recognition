#import face_model
import cv2
import numpy as np
import os
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv
from numpy import dot
from numpy.linalg import norm
import xlsxwriter

#######################################################################################################################
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--ga-model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--MaskPath', default=R'../Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_crop',help='')
parser.add_argument('--WoMaskPath', default=R'../Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop',help='')
parser.add_argument('--SavePath', default='./feature/',help='path to save feature txt file.')
parser.add_argument('--FeaturePath', default='',help='path to load feature txt file.')
parser.add_argument('--FeaturePath2', default='',help='path to load feature txt file.')
# parser.add_argument('--Protocal', default='./Protocol_verification/AFDB_new.csv',help='')
parser.add_argument('--Protocal', default='./Protocol_verification/AFDB_final_Jeff.csv',help='')
# parser.add_argument('--Method', default=['Arcface'],help='Model to test.')
parser.add_argument('--Method', default=['Arcface','Cosface','Sphereface','Arcface_mask_mouth_ori','Cosface_mask_mouth_ori','Sphereface_mask_mouth_ori'],help='Model to test.')
# parser.add_argument('--Method', default=['Arcface_mask_mouth_ori','Cosface_mask_mouth_ori','Sphereface_mask_mouth_ori'],help='Model to test.')
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

def get_cosine_txt(args,imgName1,imgName2):
    txt1=[]
    txt2=[]
    imgName1 = imgName1.split('/')[0] + '_' + imgName1.split('/')[-1].split('.')[0]
    imgName2 = imgName2.split('/')[0] + '_' + imgName2.split('/')[-1].split('.')[0]
    with open(R'{}/{}.txt'.format(args.FeaturePath,imgName1),'r') as f:
        lines = f.readlines()
        for line in lines:
            txt1.append(float(line.split('\n')[0]))
    with open(R'{}/{}.txt'.format(args.FeaturePath2,imgName2),'r') as f:
        lines = f.readlines()
        for line in lines:
            txt2.append(float(line.split('\n')[0]))
    fea1 = np.array(txt1)
    fea2 = np.array(txt2)
    similarity_score = dot(fea1, fea2)/(norm(fea1)*norm(fea2))
    return similarity_score

def main():
    x_labels = [0.0001, 0.001, 0.01, 0.1, 1]
    x_labels1 = [str(0.0001), str(0.001), str(0.01), str(0.1), str(1)]
    tpr_fpr_table = PrettyTable(['Methods'] + x_labels1)

    for method in args.Method:
        ##################Load Model######################
        args.model = './model/{}/model,0'.format(method)
        args.ga_model = args.model
        args.FeaturePath = './feature/AFDB/{}_AFDB/Mask'.format(method)
        args.FeaturePath2 = './feature/AFDB/{}_AFDB/Womask'.format(method)
        args.SavePath = './feature/AFDB/{}_AFDB'.format(method)
        if not os.path.exists(args.SavePath): os.makedirs(args.SavePath)
        # model = face_model.FaceModel(args)
        ##################################################
        
        ##################Save feature####################
        # with open(args.Protocal,'r',newline='') as f:
        #     rows = csv.reader(f)
        #     next(rows)
        #     for row in rows:
        #         feature(args,model,row[0],'Mask')
        #         feature(args,model,row[1],'WoMask')
        # print('Save {} features'.format(method))
        ##################################################

        score = []
        label = []
        ##################Complete Steps##################
        # with open(args.Protocal,'r',newline='') as f:     
        #     rows = csv.reader(f)
        #     next(rows)
        #     for row in rows:
        #         sin = get_cosine(args,model,row[0],row[1])
        #         score.append(sin)
        #         label.append(int(row[2]))
        ##################################################

        ###########Read feature ONLY from txt#############
        writer = xlsxwriter.Workbook('./result/AFDB/Analysis_{}.xlsx'.format(method))
        test = writer.add_worksheet('test')
        test.write('A1','Input')
        test.write('B1','Target')
        test.write('C1','Label')
        test.write('D1','Score')

        with open('./result/AFDB/Analysis_{}.csv'.format(method),'w',newline='') as g:
            writer2 = csv.writer(g)
            writer2.writerow(['Input','Target','Label','Score'])
            with open(args.Protocal,'r',newline='') as f:
                rows = csv.reader(f)
                next(rows)
                for idx,row in enumerate(rows):
                    sin = get_cosine_txt(args,row[0],row[1])
                    score.append(sin)
                    label.append(int(row[2]))
                    writer2.writerow([row[0],row[1],row[2],sin])
                    test.write('A{}'.format(idx+2),'=HYPERLINK("F:/Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_crop_choose/{}","{}")'.format(row[0],row[0]))
                    test.write('B{}'.format(idx+2),'=HYPERLINK("F:/Database/RMFRD/self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop_choose/{}","{}")'.format(row[1],row[1]))
                    test.write('C{}'.format(idx+2),row[2])
                    test.write('D{}'.format(idx+2),sin)
        writer.close()
                        
        
        # with open('./result/AFDB/score/Score_{}.txt'.format(method),'w',newline='') as f:
        #     for i in score:
        #         f.writelines('{}\n'.format(i))
        # with open('./result/AFDB/score/Label_{}.txt'.format(method),'w',newline='') as f:
        #     for i in label:
        #         f.writelines('{}\n'.format(i))
        #################################################

        
        fpr, tpr, t = roc_curve(label, score)
        # print(t)
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)

        tpr_fpr_row = []
        tpr_fpr_row.append(method)
        for fpr_iter in np.arange(len(x_labels)):
            _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
            tpr_fpr_row.append('%.4f' % tpr[min_index])
        tpr_fpr_table.add_row(tpr_fpr_row)

        print(roc_auc)
        if method =='Arcface':
            plt.plot(fpr, tpr, color='black', lw=1, label=(method))
        elif method == 'Cosface':
            plt.plot(fpr, tpr, color='blue', lw=1, label=(method))
        elif method == 'Sphereface':
            plt.plot(fpr, tpr, color='red', lw=1, label=(method))
        elif method == 'Arcface_mask_mouth_ori':
            plt.plot(fpr, tpr, color='black', lw=1, label=(method),linestyle="--")
        elif method == 'Cosface_mask_mouth_ori':
            plt.plot(fpr, tpr, color='blue', lw=1, label=(method),linestyle="--")
        elif method == 'Sphereface_mask_mouth_ori':
            plt.plot(fpr, tpr, color='red', lw=1, label=(method),linestyle="--")
        # elif method == 'Arcface_occlusion':
        #     plt.plot(fpr, tpr, color='black', lw=1, label=(method),linestyle=":")
        # elif method == 'Cosface_occlusion':
        #     plt.plot(fpr, tpr, color='blue', lw=1, label=(method),linestyle=":")
        # elif method == 'Sphereface_occlusion':
        #     plt.plot(fpr, tpr, color='red', lw=1, label=(method),linestyle=":")

    plt.xlim([10**-4, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xticks(x_labels) 
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    print(tpr_fpr_table)
    plt.show()

if __name__ == "__main__":
    main()