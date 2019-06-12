import numpy as np
import sklearn.metrics as metrics


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def F1_score(self, gt_image, pre_image):

        F1 = metrics.f1_score(gt_image, pre_image, average='micro')
        return F1

    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


    def per_class_iu(hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


    def label_mapping(input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int64)


    # def compute_mIoU(gt_dir, pred_dir, devkit_dir, team_name):
    #     """
    #     Compute IoU given the predicted colorized images and 
    #     """
    #     with open(join(devkit_dir, 'info.json'), 'r') as fp:
    #     info = json.load(fp)
    #     num_classes = np.int(info['classes'])
    #     print('Num classes', num_classes)
    #     name_classes = np.array(info['label'], dtype=np.str)
    #     hist = np.zeros((num_classes, num_classes))

    #     pred_imgs = [join(pred_dir, name) for name in os.listdir(pred_dir)]
    #     gt_imgs = [join(gt_dir, name) for name in os.listdir(gt_dir)]
        
    #     FF=FT=TF=TT=0

    #     for ind in range(len(gt_imgs)):

    #         print("Images:%s" % pred_imgs[ind])
    #         print("Labels:%s" % os.path.join(gt_dir, os.path.split(pred_imgs[ind])[-1]).split('.')[0] + '.png')
    #         pred = misc.imread(pred_imgs[ind], mode='L')
    #         label = misc.imread(os.path.join(gt_dir, os.path.split(pred_imgs[ind])[-1]).split('.')[0] + '.png', mode='L')
    #         pred = (pred / 255).astype(int)
    #         label = (label / 255).astype(int)
    #         print(pred.shape)
    #         print(label.shape)
    #         # #F score
    #         ff, ft, tf, tt = np.bincount((label*2+pred).reshape(-1), minlength=4)

    #         FF += ff
    #         FT += ft
    #         TF += tf
    #         TT += tt

    #         if len(label.flatten()) != len(pred.flatten()):
    #             print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
    #             continue
    #         hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
    #         if ind > 0 and ind % 10 == 0:
    #             print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
        
    #     mIoUs = per_class_iu(hist)

    #     for ind_class in range(num_classes):
    #         print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    #     R = TT / float(TT + FT)
    #     P = TT / float(TT + TF)
    #     F1 = (2*R*P)/(R+P)
    #     print(R)
    #     print(P)
    #     print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    #     print('===> F-score: ' + str(round(F1, 3)))

    #     result_dict = {'team_name':team_name ,'result': {'mIOU':str(round(np.nanmean(mIoUs) * 100, 2)), 'F-score':str(round(F_score / 2833, 3))}}
    #     with open('result.json', 'a') as f:
    #         json.dump(result_dict, f, indent=2, ensure_ascii=False)
    #         print('Write into json file')
    #     return mIoUs, F1






