import face_model
import cv2
import numpy as np
import os
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import csv
import math
import heapq

###################################
parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--ga-model', default='./model/Arcface_occlusion/model,0',help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--Method', default=['Arcface','Cosface','Sphereface','Arcface_mask_mouth_ori','Cosface_mask_mouth_ori','Sphereface_mask_mouth_ori'],help='Model to test.')
parser.add_argument('--GalleryPath', default='./feature/AFDB',help='path to load feature txt file.')
parser.add_argument('--ProbePath', default='./feature/AFDB',help='path to load feature txt file.')
parser.add_argument('--Protocal_gallery', default='./Protocol_identification/AFDB_gallery_new.csv',help='')
parser.add_argument('--Protocal_probe', default='./Protocol_identification/AFDB_probe_new.csv',help='')

args = parser.parse_args()
###################################

def get_feature_txt(args,file,path):
    fea = []
    name = file.split('/')[0] + '_' + file.split('/')[-1].split('.')[0]
    if path == 'Mask':
        with open('{}/{}.txt'.format(args.GalleryPath,name)) as f:
            lines = f.readlines()
            for line in lines:
                fea.append(line.split('\n')[0])
    elif path == 'WoMask':
        with open('{}/{}.txt'.format(args.ProbePath,name)) as f:
            lines = f.readlines()
            for line in lines:
                fea.append(line.split('\n')[0])
    return fea,file.split('/')[0]

def gen_mask(args,probe_ids, gallery_ids):
    mask = []
    for probe_id in probe_ids:
        gt_index = gallery_ids.index(probe_id)
        mask.append(gt_index)
    # print(len(mask))
    return mask

def check(probe,probe_ids,gallery,gallery_ids):
    new_probe = []
    new_probe_ids = []
    for idx,i in enumerate(probe_ids):
        if not i in gallery_ids:
            pass
        else:
            new_probe_ids.append(i)
            new_probe.append(probe[idx])

    return new_probe,new_probe_ids,gallery,gallery_ids 

def evaluation(query_feats, gallery_feats, mask):
    Fars = [0.01, 0.1]
    query_feats = np.array(query_feats).astype(np.float64)
    gallery_feats = np.array(gallery_feats).astype(np.float64)
    # print(query_feats.shape)
    # print(gallery_feats.shape)

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    # print('similarity shape', similarity.shape)
    top_inds = np.argsort(-similarity)
    # print(top_inds.shape)

    # calculate top1 
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0]
        if j == mask[i]:
            correct_num += 1
    print("top1 = {}".format(correct_num/query_num))

    # calculate top5
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:5]
        if mask[i] in j:
            correct_num += 1
    print("top5 = {}".format(correct_num/query_num))

    # calculate 10
    correct_num = 0
    for i in range(query_num):
        j = top_inds[i, 0:10]
        if mask[i] in j:
            correct_num += 1
    print("top10 = {}".format(correct_num/query_num))
    neg_pair_num = query_num * gallery_num - query_num
    # print(neg_pair_num)
    required_topk = [math.ceil(query_num * x) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0
    
    pos_sims = np.array(pos_sims)
    # print(pos_sims.shape)
    neg_sims = top_sims[np.where(top_sims > -2.0)]
    # print("neg_sims num = {}".format(len(neg_sims)))
    neg_sims = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    # print("after sorting , neg_sims num = {}".format(len(neg_sims)))
    for far, pos in zip(Fars, required_topk):
        th = neg_sims[pos-1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(far, recall, th))

def main():
    
    gallery = []
    gallery_ids = []
    probe = []    
    probe_ids = []
    
    with open(args.Protocal_gallery,'r') as f:
        rows = csv.reader(f)
        for row in rows:
            try:
                gallery.append(get_feature_txt(args,row[0],'Mask')[0])
                gallery_ids.append(get_feature_txt(args,row[0],'Mask')[1])
            except:
                continue
    with open(args.Protocal_probe,'r') as f:
        rows = csv.reader(f)
        for row in rows:
            try:
                probe.append(get_feature_txt(args,row[0],'WoMask')[0])
                probe_ids.append(get_feature_txt(args,row[0],'WoMask')[1])
            except:
                continue
    probe,probe_ids,gallery,gallery_ids = check(probe,probe_ids,gallery,gallery_ids)
    mask = gen_mask(args,probe_ids,gallery_ids)
    evaluation(probe,gallery,mask)

 
if __name__ == "__main__":
    for i in args.Method:
        args.GalleryPath = './feature/AFDB/{}_AFDB/Mask'.format(i)
        args.ProbePath = './feature/AFDB/{}_AFDB/WoMask'.format(i)
        print('*********************{}**********************'.format(i))
        main()
        print('**************************************************')