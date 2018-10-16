import os
import glob
import sys
from random import randint
import tensorflow as tf
import util3
import pandas as pd
import modelctc4
import json
import ast
import numpy as np

def convert_class_ch(indices, codes, shape):
    words = []
    word = []
    i = 0
    j=0
    for index in indices:
        if i!=index[0]:  
            words.append(word)
            i = index[0]
            word = []
        if codes[j] < 4:
            word.append(chr(codes[j]+32))
        elif codes[j] < 26:
            word.append(chr(codes[j]+34))
        elif codes[j] < 27:
            word.append(chr(codes[j]+37))
        elif codes[j] < 53:
            if codes[j] in [29, 41, 45, 48, 49, 52]:
                word.append(chr(codes[j]+ 38+ 32))
            else:
                word.append(chr(codes[j]+38))
        elif codes[j] < 56:
            word.append(chr(codes[j]+43))
        elif codes[j] < 67:
            word.append(chr(codes[j]+44))
        elif codes[j] < 70:
            word.append(chr(codes[j]+45))
        elif codes[j] < 72:
            word.append(chr(codes[j]+46))
        elif codes[j] < 74:
            word.append(chr(codes[j]+48))
        j += 1
    words.append(word)
    return words
    
def run_ctc():

    TRANSCRIPTIONFILE = os.path.curdir + '/LineValidList.csv'
    print("cur dir: ", os.path.curdir)
    ckpt = tf.train.get_checkpoint_state('./checkpoint4/')
    checkpoint_file = ckpt.model_checkpoint_path
    config_file = str('./config4.json')
    img_dir = str('./testImgPad/')
    isIAM = 1
    
    
    print("len arg: ", len(sys.argv))
    if len(sys.argv) == 1:
        print("Execution without arguments, default arguments")
        print("checkpoints_file=",checkpoint_file)
        print("config_file=", config_file)
        print("img_dir=", img_dir)
    elif len(sys.argv) == 2:       
        print("Execution without some arguments, default arguments")
        print("checkpoints_file=",checkpoint_file)
        print("config_file=", config_file)        
        img_dir = str(sys.argv[1])
    elif len(sys.argv) == 3:
        print("Execution without some arguments, default arguments")
        print("checkpoints_file=",checkpoint_file)
        print("config_file=", config_file)        
        img_dir = str(sys.argv[1])
        isIAM = int(sys.argv[2])

    elif len(sys.argv) == 4:
        print("Execution without some arguments, default arguments")
        print("config_file=", config_file)
        print("img_dir=", img_dir)
        img_dir = str(sys.argv[1])
        isIAM = int(sys.argv[2])
        checkpoint_file = str(sys.argv[3])
        
    elif len(sys.argv) == 5:
        img_dir = str(sys.argv[1])
        isIAM = int(sys.argv[2])
        checkpoint_file = str(sys.argv[3])
        config_file = str(sys.argv[4])
        
    else:
        print()
        print("ERROR")
        print("Wrong number of arguments. Execute:")
        print(">> python3 predict.py [img_dir] [isIAM] [checkpoint_file] [config_file] ")
        print("e.g. python predict.py ./img_to_predict/ 1 ./checkpoints/model.ckpt_1000 config.json ")
        exit(1)

    try:
        config = json.load(open(config_file))
    except FileNotFoundError:
        print()
        print("ERROR")
        print("No such config file : " + config_file)
        exit(1)



    BATCH_SIZE = 1
    #std_height = 300
    std_height = int(config['img_height'])
    #std_width = 1024
    std_width = int(config['img_width'])
    ctc_input_len = int(config['ctc_input_len'])
    word_len = int(config['word_len'])
    
    
    net = modelctc4.model(config)
    graph=net[0]
    X=net[1]
    Y=net[2]
    keep_prob=net[3]
    seq_len=net[4]
    optimizer=net[5]
    cost=net[6]
    ler=net[7]
    decoded=net[8]
    wer = net[9]

    #result_test = pd.DataFrame()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allocator_type = 'BFC'


    with tf.Session(graph=graph,config = sess_config) as session:
    #with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_file) 
        print("Loaded Model")
        
        predict_set = util3.dataset(img_dir, BATCH_SIZE, ctc_input_len, word_len,0, TRANSCRIPTIONFILE)
        cont = 1
        while cont > 0: 
            outputs = []
            pre_inputs, pre_seq_len, img_list = predict_set.extract_predict_data_batch(std_height,std_width, isIAM)
            # print("img list: ", img_list)
            if len(pre_inputs)>0:
                predict_feed = {X: pre_inputs, keep_prob: 1, seq_len: pre_seq_len}
                result = session.run(decoded[0], predict_feed)
                #print("result.value: ", result.values)
                #print("result.indices: ", result.indices)
                output = convert_class_ch(result.indices, result.values, result.dense_shape)
                #print("val step: ", count, "total cost: ", total_val_cost, "total ler: ", total_val_ler)
            else:
                cont = 0 
            print("outputs: ", outputs)
            for img_file, word in zip(img_list, output):
                print("image: "+img_file + "--->predict: "+ ''.join(word))
 
        
        return outputs

if __name__ == '__main__':
    run_ctc()
