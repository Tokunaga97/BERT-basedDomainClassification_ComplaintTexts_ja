# -*- coding: utf-8 -*-

from random import shuffle
import torch



def CreateDataset(tokenizer, train_path, dev_path, test_path, genre2id_dictionary, max_length=256):
    
    with open(train_path, 'r') as f:
      train_data = [line.strip().split('\t') for line in f.readlines()]
      train_data = [(text, genre2id_dictionary[genre]) for text, genre in train_data]
    
    with open(dev_path, 'r') as f:
      dev_data = [line.strip().split('\t') for line in f.readlines()]
      dev_data = [(text, genre2id_dictionary[genre]) for text, genre in dev_data]
    
    with open(test_path, 'r') as f:
      test_data = [line.strip().split('\t') for line in f.readlines()]
      test_data = [(text, genre2id_dictionary[genre]) for text, genre in test_data]
      
    shuffle(train_data)
    shuffle(dev_data)
    shuffle(test_data)
    
    
    #学習データセットを作成

    def DataCollator(batch):
      sentences_list = [text for text, _ in batch]
      genres_list = [genre for _, genre in batch]
    
      tokenized_senentences = tokenizer(sentences_list, padding='max_length', truncation=True, max_length=max_length)
    
      output_dic = {}
    
      output_dic['input_ids'] = torch.tensor(tokenized_senentences['input_ids'])
      output_dic['attention_mask'] = torch.tensor(tokenized_senentences['attention_mask'])
      output_dic['labels'] = torch.tensor([genres_list[i] for i in range(len(genres_list))])
    
      return output_dic
    
    
    train_datasets = train_data
    dev_datasets = dev_data
    test_datasets = test_data
    
    data_collator_fn = DataCollator
    
    return train_datasets, dev_datasets, test_datasets, data_collator_fn
  
    

