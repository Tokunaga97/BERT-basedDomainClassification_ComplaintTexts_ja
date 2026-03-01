# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from dataset import CreateDataset
from bert_test import Test

max_length = 256
batch_size = 16

load_dir_path = '../loaded_data'

train_path = r'../corpus/train.tsv'
dev_path = r'../corpus/dev.tsv'
test_path = r'../corpus/test.tsv'
genre_path = r'../corpus/genres.tsv'

with open(genre_path, 'r') as f:
    genres = [genre.strip() for genre in f.readlines()]

genre2id_dictionary = {g:i for i, g in enumerate(genres)}
id2genre_dictionary = {i:g for i, g in enumerate(genres)}

tokenizer = AutoTokenizer.from_pretrained(load_dir_path)
model = BertForSequenceClassification.from_pretrained(load_dir_path, num_labels=len(genre2id_dictionary))

train_datasets, dev_datasets, test_datasets, data_collator_fn = CreateDataset(tokenizer, train_path, dev_path, test_path, genre2id_dictionary, max_length=max_length)



training_args = TrainingArguments(
    output_dir= load_dir_path,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    #evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=1e-4,
    weight_decay=0.01,
    eval_steps=10,
    save_steps=20,
    warmup_steps=10,
    gradient_accumulation_steps=4,
    save_total_limit=5,
    prediction_loss_only=True,
    fp16=True,
    fp16_full_eval=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=2
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator_fn,
    train_dataset=train_datasets,
    eval_dataset=dev_datasets
)

torch.cuda.empty_cache()
#ファインチューニング
trainer.train()
#保存
trainer.save_model(load_dir_path)

Test(tokenizer, model, id2genre_dictionary)