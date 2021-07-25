import os
import cv2
import numpy as np
import pandas as pd



img_path = './sample'
LM_root = './landmark'
LM_file = 'test.csv'
save_root = './mask'

lmrk = pd.read_csv('{}/{}'.format(LM_root, LM_file))


if not os.path.exists(save_root):
    os.makedirs(save_root)


def frontal(img):
    points = np.array([[int(landmarks[1,0]), int(landmarks[1,1])],
                       [int(landmarks[2, 0]), int(landmarks[2, 1])],
                       [int(landmarks[3,0]), int(landmarks[3,1])],
                       [int(landmarks[4,0]), int(landmarks[4,1])],
                       [int(landmarks[5, 0]), int(landmarks[5, 1])],
                       [int(landmarks[6, 0]), int(landmarks[6, 1])],
                       [int(landmarks[7, 0]), int(landmarks[7, 1])],
                       [int(landmarks[8,0]), int(landmarks[8,1])],
                       [int(landmarks[9, 0]), int(landmarks[9, 1])],
                       [int(landmarks[10, 0]), int(landmarks[10, 1])],
                       [int(landmarks[11, 0]), int(landmarks[11, 1])],
                       [int(landmarks[12, 0]), int(landmarks[12, 1])],
                       [int(landmarks[13, 0]), int(landmarks[13, 1])],
                       [int(landmarks[14,0]), int(landmarks[14,1])],
                       [int(landmarks[15, 0]), int(landmarks[15, 1])],
                       [int(landmarks[28,0]), int(landmarks[28,1])]])
    img = cv2.fillConvexPoly(img, points, (0, 0, 0))

    return img


def profile(img):
    points = np.array([[int(landmarks[7,0]), int(landmarks[7,1])],
                       [int(landmarks[8, 0]), int(landmarks[8, 1])],
                       [int(landmarks[9, 0]), int(landmarks[9, 1])],
                       [int(landmarks[10, 0]), int(landmarks[10, 1])],
                       [int(landmarks[11, 0]), int(landmarks[11, 1])],
                       [int(landmarks[12, 0]), int(landmarks[12, 1])],
                       [int(landmarks[13, 0]), int(landmarks[13, 1])],
                       [int(landmarks[14,0]), int(landmarks[14,1])],
                       [int(landmarks[15, 0]), int(landmarks[15, 1])],
                       [int(landmarks[28, 0]), int(landmarks[28, 1])],
                       [int(landmarks[29,0]), int(landmarks[29,1])],
                       [int(landmarks[30, 0]), int(landmarks[30, 1])],
                       ])
    img = cv2.fillConvexPoly(img, points, (0, 0, 0))


    return img

def blur(img):

    img_copy = img.copy()
    img_copy[:, :, :] = 255


    line_color = (0, 0, 0)
    line_width = 1
    if face_type.split('_')[0] == '0':

        for n in range(1, 15):
            cv2.line(img_copy, (int(landmarks[n, 0]), int(landmarks[n, 1])),
                     (int(landmarks[n + 1, 0]), int(landmarks[n + 1, 1])), line_color, line_width)
        cv2.line(img_copy, (int(float(landmarks[15, 0])), int(float(landmarks[15, 1]))),
                 (int(float(landmarks[28, 0])), int(float(landmarks[28, 1]))), line_color, line_width)
        cv2.line(img_copy, (int(float(landmarks[28, 0])), int(float(landmarks[28, 1]))),
                 (int(float(landmarks[1, 0])), int(float(landmarks[1, 1]))), line_color, line_width)
    else:
        for n in range(7, 15):
            cv2.line(img_copy, (int(landmarks[n, 0]), int(landmarks[n, 1])),
                     (int(landmarks[n + 1, 0]), int(landmarks[n + 1, 1])), line_color, line_width)
        cv2.line(img_copy, (int(float(landmarks[15, 0])), int(float(landmarks[15, 1]))),
                 (int(float(landmarks[28, 0])), int(float(landmarks[28, 1]))), line_color, line_width)
        cv2.line(img_copy, (int(float(landmarks[28, 0])), int(float(landmarks[28, 1]))),
                 (int(float(landmarks[29, 0])), int(float(landmarks[29, 1]))), line_color, line_width)
        cv2.line(img_copy, (int(float(landmarks[29, 0])), int(float(landmarks[29, 1]))),
                 (int(float(landmarks[30, 0])), int(float(landmarks[30, 1]))), line_color, line_width)
        cv2.line(img_copy, (int(float(landmarks[30, 0])), int(float(landmarks[30, 1]))),
                 (int(float(landmarks[7, 0])), int(float(landmarks[7, 1]))), line_color, line_width)



    kernel_size = (3, 3)
    sigma = 0.5

    for i in range(112):
        for j in range(112):

            if img_copy[i, j, 0] == 0:

                img[i-1:i+2,j-1:j+2,0] = cv2.GaussianBlur(img[i-1:i+2,j-1:j+2,0], kernel_size, sigma)
                img[i-1:i+2,j-1:j+2,1] = cv2.GaussianBlur(img[i-1:i+2,j-1:j+2,1], kernel_size,
                                                                      sigma)
                img[i-1:i+2,j-1:j+2,2] = cv2.GaussianBlur(img[i-1:i+2,j-1:j+2,2], kernel_size,
                                                                      sigma)

    return img



for idx in range(0, len(lmrk)):

    print('Processing [{}/{}] images ...'.format(idx, len(lmrk)))
    sub_f = lmrk.sub_folder[idx]
    id_name = lmrk.image_name[idx]
    face_type = lmrk.face_type[idx]
    landmarks = lmrk.iloc[idx, 3:].values
    landmarks = landmarks.astype('float').reshape(-1, 2)
    img_root = '{}/{}/{}'.format(img_path, sub_f[1:],id_name)


    img = cv2.imread(img_root)

    if face_type.split('_')[0] == '0':
        img_final = frontal(img)
        if face_type.split('_')[1] == '1':
            img_final = blur(img)
    else:
        img_final = profile(img)
        if face_type.split('_')[1] == '1':
            img_final = blur(img)

    cv2.imshow('img_final',img_final)
    cv2.waitKey(0)
    save_path = '{}/{}'.format(save_root, id_name)
    cv2.imwrite(save_path,img_final)





