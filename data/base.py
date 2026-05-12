import os
import logging
import csv
import copy
import numpy as np
import torch
import torch.nn as nn

from .MMDataset import MMDataset
from .text_pre import get_t_data
from .mm_pre import get_v_a_data
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)
        self.mm_data = get_data(args, self.logger) 
        self.labels_weight = get_labels_weight(args, benchmarks)


        
def get_data(args, logger):
    

    data_path = os.path.join(args.data_path, args.dataset)

    bm = benchmarks[args.dataset]
    

    label_list = copy.deepcopy(bm["intent_labels"])
    logger.info('Lists of intent labels are: %s', str(label_list))  
      
    args.num_labels = len(label_list)  
    args.text_feat_dim = bm['feat_dims']['text']
    args.video_feat_dim = bm['feat_dims']['video']
    args.audio_feat_dim = bm['feat_dims']['audio']
    args.label_len = bm['label_len']
    logger.info('In-distribution data preparation...')
    
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path,"train.tsv"), args.data_mode)

    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'dev.tsv'), args.data_mode)

    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'test.tsv'), args.data_mode)

    args.num_train_examples = len(train_data_index) 
    
    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
        'bm': bm,
    }

    text_seq_len: object
    data_args['max_seq_len'] = args.text_seq_len = bm['max_seq_lengths']['text']
    text_data, cons_text_feats, condition_idx = get_t_data(args, data_args)
    
    
    video_feats_path = os.path.join(data_path, 'video_feats.pkl')
    video_feats_data_args = {
        'data_path': video_feats_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
    }
    video_feats_data_args['max_seq_len'] = args.video_seq_len = bm['max_seq_lengths']['video_feats']
    
    video_feats_data = get_v_a_data(video_feats_data_args, video_feats_path)

    audio_feats_path = os.path.join(data_path, 'audio_feats.pkl')
    audio_feats_data_args = {
        'data_path': audio_feats_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
    }
    audio_feats_data_args['max_seq_len'] = args.audio_seq_len = bm['max_seq_lengths']['audio_feats']
    
    audio_feats_data = get_v_a_data(audio_feats_data_args, audio_feats_path)
    

    train_data = MMDataset(train_label_ids, text_data['train'], video_feats_data['train'], audio_feats_data['train'], cons_text_feats['train'], condition_idx['train'])
    dev_data = MMDataset(dev_label_ids, text_data['dev'], video_feats_data['dev'], audio_feats_data['dev'], cons_text_feats['dev'], condition_idx['dev']) 
    test_data = MMDataset(test_label_ids, text_data['test'], video_feats_data['test'], audio_feats_data['test'], cons_text_feats['test'], condition_idx['test'])

    data = {'train': train_data, 'dev': dev_data, 'test': test_data}     
    

    return data
    

def get_indexes_annotations(args, bm, label_list, read_file_path, data_mode):

    if args.dataset in ['MELD']:
        label_map = {}
        for i, label in enumerate(label_list):
            label_map[label] = i

    else:
        label_map = bm['labels_map']


    with open(read_file_path, 'r', encoding='utf-8', newline='') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            if args.dataset in ['MIntRec']:  
                index = '_'.join([line[0], line[1], line[2]])
                indexes.append(index)
                label_id = label_map[line[4]]

            elif args.dataset in ['MELD']:
                index = '_'.join([line[0], line[1]])
                indexes.append(index)
                
                label_id = label_map[bm['label_maps'][line[3]]]

            elif args.dataset in ['MIntRec2']:
                index = 'dia{}_utt{}'.format(line[0], line[1])
                indexes.append(index)
                
                label_id = label_map[line[3]]
            
            label_ids.append(label_id)

    
    return indexes, label_ids

def get_labels_weight(args, benchmarks):
    bm = benchmarks[args.dataset]

    label_weight = bm['labels_weight']
    weights = {label: 1.0 / weight for label, weight in label_weight.items()}
    total = sum(weights.values())
    normalized_weights = {label: weight / total for label, weight in weights.items()}

    labels_map = bm['labels_map']
    weight_tensor = torch.tensor([normalized_weights[label] for label in sorted(labels_map, key=labels_map.get)])

    return weight_tensor

