import os
import sys
from PIL import Image
import pandas as pd
import numpy as np
import glob
import cv2
import math
import random
import preprocessImg4
import csv

class dataset:
    def __init__(self, path, batch_size, ctc_input_len, word_len, repeat, transcriptionsFile):
        #self.path = path
        self.__batch_size = batch_size
        self.__ctc_input_len = ctc_input_len
        self.__word_len = word_len
        self.__repeat = repeat
        self.__index = 0
        self.__continueExtractBatch = True
        #self.__level = level
        self.__sample_set = self.get_sample_filenames(path)
        if repeat==1:
            random.shuffle(self.__sample_set)
        self.__transcriptions_set = self.get_sample_transcriptions(transcriptionsFile)

        
    def get_sample_transcriptions(self, file):
        # return a dictionary {filename, text in the image}
        transcriptions={}
        with open(file) as f:
            f_csv = csv.reader(f)
            count = 0
            header = next(f_csv)
            isSecondQuote = False
            for row in f_csv:                                   
                transcriptions[row[0]] = row[1]
        return transcriptions
    
    def get_transcriptions_for_batch(self, batch_set):
        lines = []
        for file in batch_set:
            fileName = file.split(os.sep)[-1]
            #print("first spit: ", word)
            fileName = fileName.split('.')[-2]
            lines.append(self.__transcriptions_set[fileName])
        return lines
        
    def convert_character_to_label_class(self, char):
        
        if char in ['c', 'o', 's', 'v', 'w', 'z']:
            ch = char.upper()
            # print(ch)
        else:
            ch = char
        if ord(ch) > 31 and ord(ch) < 36:
            return(ord(ch)-32)
        if ord(ch) >37 and ord(ch) < 60:
            return(ord(ch)-34)
        if ord(ch) >62 and ord(ch) < 64:    
            return(ord(ch)-37)
        if ord(ch) > 64 and ord(ch) < 91:
            return(ord(ch)-38)
        if ord(ch) > 95 and ord(ch) < 99:
            return(ord(ch)-43)
        if ord(ch) > 99 and ord(ch) < 111:  ##字母c的ascii码是99 字母o的ascii码是111
            return(ord(ch) - 44)
        if ord(ch) > 111 and ord(ch) < 115: #字母s的ascii码是115
            return(ord(ch) - 45)
        if ord(ch) > 115 and ord(ch) < 118: #字母v的ascii码是118
            return(ord(ch) - 46)
        if ord(ch) > 119 and ord(ch) < 122: #字母w的ascii码是119, #字母z的ascii码是122
            return(ord(ch) - 48)
        else:
            print("error")
            

        
    def extract_data_batch(self):

        """
            Return batch of images and labels in a random way to train model 

            return:

              - batchx: Tensor with images as matrices
                (Array of Floats: [batch_size, height, width, 1])
              - sparse: SparseTensor with labels (SparseTensor: indice,values,shape)
              - transcriptions: Arraywith labels of "batchx". (Array de Strings: [batch_size])
              - seq_len: Array with input length for CTC, "ctc_input_len". (Array of Ints: [batch_size])
        """
        
        batchx = []
        transcriptions = []
        indice = []
        seq_len=[]
        values = []
        i = 0
        
        if not self.__continueExtractBatch:
            return batchx, None, transcriptions, seq_len
            
        # For training mode, shuffle the dataset after each round of walking through all samples
        #if self.__repeat and self.__index +self.__batch_size > len(self.__sample_set):
        #    random.shuffle(self.__sample_set)
        #    self.__index = 0  
        head = self.__index % len(self.__sample_set)
        tail = (self.__index +self.__batch_size)% len(self.__sample_set)
        if self.__repeat:
            if tail < head:
                batch_set = self.__sample_set[0:tail]+self.__sample_set[head: len(self.__sample_set)]
                random.shuffle(self.__sample_set)
                self.__index = 0
            else:
                batch_set = self.__sample_set[head:tail]
                self.__index = tail + self.__batch_size
        else:
            if tail < head:  
                batch_set = self.__sample_set[head: len(self.__sample_set)]
                self.__continueExtractBatch = False
            else:
                batch_set = self.__sample_set[head:tail]
                self.__index = tail + self.__batch_size
            
        #batch_set = self.__sample_set[self.__index:self.__index+self.__batch_size]
        lines = self.get_transcriptions_for_batch(batch_set)
        i =0    
        for file, line in zip(batch_set, lines):
            if len(line)==0:
                print("word is empty for image ", file, "return empyt batch")
                return None, None, transcriptions, seq_len
            else:
                img = cv2.imread(file,0)

                height = img.shape[0]
                width = img.shape[1]
                if (height != 320) or (width != 3200):
                    print("image size error: ", file, height, width)
                    return None, None, transcriptions, seq_len
                result=img.reshape(height, width,1)
                result = result/255.0
                batchx.append(result)

                # extract labels, and create indice for sparse tensor 
                j = 0
                for ch in list(str(line)):
                    if j>self.__ctc_input_len:
                        print("Error: line", line, " length", j, " > ctc_input_len", self.__ctc_input_len, " return empty batch")
                        return None, None, transcriptions, seq_len 
                    if (ord(ch)<32) and (ord(ch)>122): 
                        print("Error: character is out of scope: ", ch) 
                        return None, None, transcriptions, seq_len  
                    else: 
                        label_code = self.convert_character_to_label_class(ch)
                        if label_code > 73:
                            print("error label code > 74: ", label_code, " ->" , ch)
                            return None, None, transcriptions, seq_len 
                        else:
                            values.append(self.convert_character_to_label_class(ch)) 
                            indice.append([i,j])                   
                    j = j + 1                           
                transcriptions.append(line)
                
                # seq_len: tensor shape (batch_size), value: [_ctc_input_len, _ctc_input_len,..._ctc_input_len]
                seq_len.append(self.__ctc_input_len)
                i +=1
        self.__index += self.__batch_size
        batchx = np.stack(batchx, axis=0)   
        shape=[self.__batch_size,self.__word_len]
        sparse=indice,values,shape
        #print ("labels values: ", values) 
        #print ('batch set: ', batch_set)
        return batchx, sparse, transcriptions, seq_len
        
        
    def extract_predict_data_batch(self, std_height, std_width, isIAM):

        batchx = []
        indice = []
        seq_len=[]
 
        i = 0
        if self.__index >= len(self.__sample_set):           
            return batchx, seq_len, []
        if self.__index + self.__batch_size > len(self.__sample_set):  
            batch_set = self.__sample_set[self.__index:]           
        else:
            batch_set = self.__sample_set[self.__index:self.__index+self.__batch_size]

        for file in batch_set:
            if isIAM == 1:
                img = cv2.imread(file,0)
            else:
                img = preprocessImg4.normalizeImg(filename=file, img=None, stdRow=std_height, stdCol=std_width)
            
            
            height = img.shape[0]
            width = img.shape[1]
  
            result=img.reshape(height, width,1)
            result = result/255.0
            batchx.append(result)
            seq_len.append(self.__ctc_input_len)
            i +=1
        self.__index += len(batch_set)
        batchx = np.stack(batchx, axis=0)   
 
        return batchx, seq_len, batch_set

    def get_sample_filenames(self, directory):

        file_set_png = sorted(glob.glob(directory+'*.png'))
        file_set_jpg = sorted(glob.glob(directory+'*.jpg'))
        file_set = file_set_png + file_set_jpg
           # if len(file_set) == 0:
           #     file_set=sorted(glob.glob(directory+'*.jpg'))
 
        return file_set
    


