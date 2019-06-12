import numpy as np
import multiprocessing
import os
from PIL import Image

def do_python_eval():
    MODEL_NUM_CLASSES = 2
    categories = ['foreground','background']
    #predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
    #gt_folder = self.seg_dir
    predict_folder = os.listdir('./results/')
    #gt_folder = os.listdir('/home/xupeihan/data/Fashionista/testLabel/')
    path = '/home/xupeihan/data/voc_detection/person_label'
    gt_folder = os.listdir('/home/xupeihan/data/voc_detection/person_label/')


    TP = []
    P = []
    T = []
    for i in range(MODEL_NUM_CLASSES):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    
    def compare(start,step,TP,P,T):
        for idx in range(start,len(predict_folder),step):
            print('%d/%d'%(idx,len( predict_folder)))
            name =  predict_folder[idx]
            #predict_file = os.path.join(predict_folder,'%s.png'%name)
            predict_file = os.path.join('./results/',name)
            gt_file = os.path.join(path,name[:-3]+'png')
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
            gt = np.array(Image.open(gt_file))
            gt[gt == 255] = 1
            predict[predict == 255] = 1
            cal = gt<255
            mask = (predict==gt) * cal
        
            for i in range(MODEL_NUM_CLASSES):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    for i in range(MODEL_NUM_CLASSES):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
    for i in range(MODEL_NUM_CLASSES):
        if i == 0:
            print('%11s:%7.3f%%'%('backbound',IoU[i]*100),end='\t')
        else:
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i-1],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i-1],IoU[i]*100))
                
    miou = np.mean(np.array(IoU))
    print('\n======================================================')
    print('%11s:%7.3f%%'%('mIoU',miou*100))    

    return miou*100
do_python_eval()