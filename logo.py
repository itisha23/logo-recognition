#!/usr/bin/env python
# coding: utf-8

# In[16]:


import cv2
import glob
import numpy as np
import pandas as pd
import scikitplot as skplt
from scipy import interp
from itertools import cycle
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score, auc

class BOV_SIFT:
    
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        self.trainImageCount = 0
        self.testImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.clf = SVC()
        self.kmeans_obj = KMeans(n_clusters = n_clusters)
        self.mega_histogram = None
        self.kmeans_ret = None
        
    def developVocabulary(self, kmeans_ret=None):
        
        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(self.trainImageCount)])
        old_cnt=0
        for i in range(self.trainImageCount):
            le = len(self.descriptor_list[i])
            for j in range(le):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_cnt+j]
                else:
                    idx = kmeans_ret[old_cnt+j]
                self.mega_histogram[i][idx]+=1
            old_cnt+=le
        print("Histogram Generated")
    
        
    def standardize(self, std=None):
        
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
            
        else:
            print("STD not None. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)
            
            
    def trainModel(self):
        print("Features extraction using SIFT")
        imlist = {}
        count = 0

        path = "C:/Users/ravina-singh/Desktop/Project/Train"
        for each in glob.glob(path + "/*"):
            word = each.split("/")[-1].split("\\")[-1]
            imlist[word] = []
            for imagefile in glob.glob(path+"/"+word+"/*"):
                im = cv2.imread(imagefile,0)
                imlist[word].append(im)
                count = count+1
    
        self.trainImageCount = count
        #print(self.trainImageCount)
        label_count = 0
        
        sum=0
        for word, imglist in imlist.items():
            
            self.name_dict[str(label_count)] = word
            
            for im in imglist:
                self.train_labels = np.append(self.train_labels,label_count)
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(im,None)
                sum=sum+des.shape[0]
                self.descriptor_list.append(des)
                
            label_count +=1
            
        #print(self.name_dict)
        #print("hello",sum)
        #print(len(self.descriptor_list[0]))
        #print(self.descriptor_list[0].shape)
        
        #CLUSTERING 
        
        vStack = np.array(self.descriptor_list[0])
        for remaining in self.descriptor_list[1:]:
            vStack = np.vstack((vStack,remaining)) 
        #print(len(vStack))
        
        self.kmeans_ret = self.kmeans_obj.fit_predict(vStack)
        #print(kmeans_ret.shape)
        
        #GENERATING VOCABULARY HISTOGRAM
        self.developVocabulary()
        
        #STANDARIZING
        self.standardize()
        #It is required to normalize the distribution wrt the sample size and features. 
        #If not normalized, the classifier may become biased due to steep variances.
        
        #TRAINING SVM
        print("Training SVM")
        self.clf.fit(self.mega_histogram, self.train_labels)
        print("Training Completed")
        
    def predict(self, des):
        predictions = self.clf.predict(des)
        
        return predictions
        
    def testModel(self):
        imlist = {}
        count = 0
        y_true = []
        y_pred = []

        path = "C:/Users/ravina-singh/Desktop/Project/Test"
        for each in glob.glob(path + "/*"):
            word = each.split("/")[-1].split("\\")[-1]
            imlist[word] = []
            for imagefile in glob.glob(path+"/"+word+"/*"):
                im = cv2.imread(imagefile,0)
                imlist[word].append(im)
                count = count+1
        
        self.testImageCount = count
        
        predictions = []
        
        label_count = 0
        for word, imglist in imlist.items():
            
            for im in imglist:
                #cl = self.recognize(im)
                
                #RECOGNITION
                y_true.append(label_count)
                sift = cv2.xfeatures2d.SIFT_create()
                kp, des = sift.detectAndCompute(im,None)
                
                test_ret = self.kmeans_obj.predict(des)
                vocab = np.array([[0 for i in range(self.n_clusters)]])
                
                for each in test_ret:
                    vocab[0][each]+= 1
                
                vocab = self.scale.transform(vocab)
                cl = self.clf.predict(vocab)
                
                predictions.append({'Image':im, 'Class':cl, 'Object_name':self.name_dict[str(int(cl[0]))]})
                y_pred.append(int(cl[0]))
            label_count+= 1
                
        #print(predictions)
        #print(y_true)
        #print(y_pred)
        
        print(accuracy_score(y_true,y_pred))
        print(classification_report(y_true,y_pred))
        print(f1_score(y_true, y_pred,average='macro'))
        print(confusion_matrix(y_true, y_pred))
        
        n_classes = 5
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_true))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        lw=2
        plt.figure(figsize=(8,5))
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
                 color='green', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic of SVM using SIFT fetaures')
        plt.legend(loc="lower right")
        plt.show()
        
        for each in predictions:
            
            plt.imshow(cv2.cvtColor(each['Image'],cv2.COLOR_GRAY2RGB))
            plt.title(each['Object_name'])
            plt.show()
        


# In[17]:


class BOV_SURF:
     
    def __init__(self,n_clusters):
        self.n_clusters = n_clusters
        self.trainImageCount = 0
        self.testImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.clf = SVC()
        self.kmeans_obj = KMeans(n_clusters = n_clusters)
        self.mega_histogram = None
        self.kmeans_ret = None
        
    def developVocabulary(self, kmeans_ret=None):
        
        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(self.trainImageCount)])
        old_cnt=0
        for i in range(self.trainImageCount):
            le = len(self.descriptor_list[i])
            for j in range(le):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_cnt+j]
                else:
                    idx = kmeans_ret[old_cnt+j]
                self.mega_histogram[i][idx]+=1
            old_cnt+=le
        print("Histogram Generated")
    
        
    def standardize(self, std=None):
        
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
            
        else:
            print("STD not None. External STD supplied")
            self.mega_histogram = std.transform(self.mega_histogram)
            
            
    def trainModel(self):
        
        imlist = {}
        count = 0

        path = "C:/Users/ravina-singh/Desktop/Project/Train"
        for each in glob.glob(path + "/*"):
            word = each.split("/")[-1].split("\\")[-1]
            imlist[word] = []
            for imagefile in glob.glob(path+"/"+word+"/*"):
                im = cv2.imread(imagefile,0)
                imlist[word].append(im)
                count = count+1
    
        self.trainImageCount = count
        #print(self.trainImageCount)
        label_count = 0
        
        sum=0
        for word, imglist in imlist.items():
            
            self.name_dict[str(label_count)] = word
            
            for im in imglist:
                self.train_labels = np.append(self.train_labels,label_count)
                surf = cv2.xfeatures2d.SURF_create()
                kp, des = surf.detectAndCompute(im,None)
                sum=sum+des.shape[0]
                self.descriptor_list.append(des)
                
            label_count +=1
            
        #print(self.name_dict)
        #print("hello",sum)
        #print(len(self.descriptor_list[0]))
        #print(self.descriptor_list[0].shape)
        
        #CLUSTERING 
        
        vStack = np.array(self.descriptor_list[0])
        for remaining in self.descriptor_list[1:]:
            vStack = np.vstack((vStack,remaining)) 
        #print(len(vStack))
        
        self.kmeans_ret = self.kmeans_obj.fit_predict(vStack)
        #print(kmeans_ret.shape)
        
        #GENERATING VOCABULARY HISTOGRAM
        self.developVocabulary()
        
        #STANDARIZING
        self.standardize()
        '''It is required to normalize the distribution wrt the sample size and features. 
        If not normalized, the classifier may become biased due to steep variances.'''
        
        #TRAINING SVM
        print("Training SVM")
        self.clf.fit(self.mega_histogram, self.train_labels)
        print("Training Completed")
        
    def predict(self, des):
        predictions = self.clf.predict(des)
        
        return predictions
        
    def testModel(self):
        print("Features extraction using SURF")
        
        imlist = {}
        count = 0
        y_true = []
        y_pred = []

        path = "C:/Users/ravina-singh/Desktop/Project/Test"
        for each in glob.glob(path + "/*"):
            word = each.split("/")[-1].split("\\")[-1]
            imlist[word] = []
            for imagefile in glob.glob(path+"/"+word+"/*"):
                im = cv2.imread(imagefile,0)
                imlist[word].append(im)
                count = count+1
        
        self.testImageCount = count
        
        predictions = []
        
        label_count = 0
        for word, imglist in imlist.items():
            
            for im in imglist:
                #cl = self.recognize(im)
                
                #RECOGNITION
                y_true.append(label_count)
                surf = cv2.xfeatures2d.SURF_create()
                kp, des = surf.detectAndCompute(im,None)
                
                test_ret = self.kmeans_obj.predict(des)
                vocab = np.array([[0 for i in range(self.n_clusters)]])
                
                for each in test_ret:
                    vocab[0][each]+= 1
                
                vocab = self.scale.transform(vocab)
                cl = self.clf.predict(vocab)
                
                predictions.append({'Image':im, 'Class':cl, 'Object_name':self.name_dict[str(int(cl[0]))]})
                y_pred.append(int(cl[0]))
            label_count+= 1
                
        #print(predictions)
        #print(y_true)
        #print(y_pred)
        
        print(accuracy_score(y_true,y_pred))
        print(classification_report(y_true,y_pred))
        print(f1_score(y_true, y_pred,average='macro'))
        print(confusion_matrix(y_true, y_pred))
        
        n_classes = 5
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_true))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        lw=2
        plt.figure(figsize=(8,5))
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
                 color='green', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic of SVM using SURF fetaures')
        plt.legend(loc="lower right")
        plt.show()
        
        for each in predictions:
            
            plt.imshow(cv2.cvtColor(each['Image'],cv2.COLOR_GRAY2RGB))
            plt.title(each['Object_name'])
            plt.show()


# In[18]:


if __name__=='__main__':
    bov_sift = BOV_SIFT(n_clusters=58)
    bov_sift.trainModel()
    bov_sift.testModel()
    
    bov_surf = BOV_SURF(n_clusters=58)
    bov_surf.trainModel()
    bov_surf.testModel()


# In[ ]:





# In[ ]:




