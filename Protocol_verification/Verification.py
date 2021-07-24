import os
import random
import csv


def main():
    WoMaskPath = R'F:\Database\RMFRD\self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop_choose'
    MaskPath = R'F:\Database\RMFRD\self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_crop_pose'
    Mask = []
    WoMask = []

    a = []
    # with open (R'F:\CVPRW\Protocol_identification/AFDB_gallery_new4.csv','r') as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         a.append(i.split('/')[0])
    for i in os.listdir(MaskPath):
        if not i in a:
            a.append(i)
        for j in os.listdir('{}/{}'.format(MaskPath,i)):
            Mask.append('{}/{}'.format(i,j))
    for i in os.listdir(WoMaskPath):
        if i in a:
            for j in os.listdir('{}/{}'.format(WoMaskPath,i)):
                # if os.path.exists(R'F:\Database\RMFRD\self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop_3/{}/{}'.format(i,j)):
                WoMask.append('{}/{}'.format(i,j))
    ## Mask 隨機挑選一張
    # for i in os.listdir(MaskPath):
    #     mask_sample = random.sample(os.listdir('{}/{}'.format(MaskPath,i)), 1)
    #     try:
    #         womask_sample = random.sample(os.listdir('{}/{}'.format(WoMaskPath,i)), 10)
    #     except FileNotFoundError:
    #         continue
    #     except ValueError:
    #         num = len(os.listdir('{}/{}'.format(WoMaskPath,i)))
    #         print(num)
    #         womask_sample = random.sample(os.listdir('{}/{}'.format(WoMaskPath,i)), num)
    #     Mask.append('{}/{}'.format(i,mask_sample[0]))
    #     for j in womask_sample:
    #             WoMask.append('{}/{}'.format(i,j))
    print('hi')
    id = ''
    with open ('./AFDB_0829_all.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Input','Target','Label'])
        for i in Mask:
            num = 0 
            for j in WoMask:                               
                if i.split('/')[0] == j.split('/')[0]:
                    writer.writerow([i,j,'1'])
                else:
                    # if num > 10:
                    #     continue
                    # else:
                    if j.split('/')[0] != id:
                        writer.writerow([i,j,'0'])
                        num+=1
                        id = j.split('/')[0]
    print()

if __name__ == "__main__":
    main()