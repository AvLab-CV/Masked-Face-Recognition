import os
import csv
import random
import glob

def main():
    ProbePath = R'F:\Database\RMFRD\self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop_choose'
    GalleryPath = R'F:\Database\RMFRD\self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset_crop_pose_all'
    probe = []
    gallery = []


    a = []
    # with open('./AFDB_gallery_new.csv','r') as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         a.append(i.split('/')[0])
    for i in os.listdir(GalleryPath):
        if not i in a:
            a.append(i)
        gallery.append('{}/{}'.format(i,random.sample(os.listdir('{}/{}'.format(GalleryPath,i)), 1)[0]))
        # for j in os.listdir('{}/{}'.format(GalleryPath,i)):
        #     gallery.append('{}/{}'.format(i,j))
        #     gallery.append(random.sample(os.listdir('{}/{}'.format(GalleryPath,i)), 1))
    # for i in os.listdir(ProbePath):
    #     if i in a:
    #         try:
    #             probe_sample = random.sample(os.listdir('{}/{}'.format(ProbePath,i)), 10)
    #             for j in probe_sample:
    #                 probe.append('{}/{}'.format(i,j))
    #         except ValueError:
    #             num = len(os.listdir('{}/{}'.format(ProbePath,i)))
    #             print(num)
    #             probe_sample = random.sample(os.listdir('{}/{}'.format(ProbePath,i)), num)
    #             for j in probe_sample:
    #                 probe.append('{}/{}'.format(i,j))
            # for j in os.listdir('{}/{}'.format(ProbePath,i)):
                # if os.path.exists(R'F:\Database\RMFRD\self-built-masked-face-recognition-dataset/AFDB_face_dataset_crop_3/{}/{}'.format(i,j)):
                # probe.append('{}/{}'.format(i,j))


    # for i in os.listdir(GalleryPath):
    #     gallery_sample = random.sample(os.listdir('{}/{}'.format(GalleryPath,i)), 1)
    #     try:
    #         probe_sample = random.sample(os.listdir('{}/{}'.format(ProbePath,i)), 15)
    #     except FileNotFoundError:
    #         continue
    #     except ValueError:
    #         num = len(os.listdir('{}/{}'.format(ProbePath,i)))
    #         print(num)
    #         probe_sample = random.sample(os.listdir('{}/{}'.format(ProbePath,i)), num)
    #     gallery.append('{}/{}'.format(i,gallery_sample[0]))
    #     for j in probe_sample:
    #         probe.append('{}/{}'.format(i,j))
        
    # with open('./AFDB_probe_0831.csv','w',newline='') as f:
    #     writer = csv.writer(f)
    #     for i in probe:
    #         writer.writerow([i])
    with open('./AFDB_gallery_0831_2.csv','w',newline='') as f:
        writer = csv.writer(f)
        for i in gallery:
            writer.writerow([i])
    
if __name__ == "__main__":
    main()