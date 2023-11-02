import datasets

from model import SBA
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
import torch
import random
from tqdm import tqdm
import numpy as np

from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import os
import inspect

import argparse


class SBADataset(Dataset):
    def load_raw_data(self, path):
        raw_data = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                raw_data.append(line.split('\n')[0])
        return raw_data

    def __init__(self, data_dir, split_name):
        data_path = os.path.join(data_dir, split_name + '.target')
        self.data = self.load_raw_data(data_path)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx], return_tensors='pt',
                                padding='max_length', truncation=True, max_length=128)
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        return {"input_ids": input_ids,
                "attention_mask": attention_mask}


class SBATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        loss = model(input_ids, attention_mask, is_train=True)[1]['loss']
        return loss



def main(args):
    random.seed(42)
    torch.manual_seed(42)

    train_dataset = SBADataset(args.data_dir, 'train')
    val_dataset = SBADataset(args.data_dir, 'val')
    # train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    sba_config = RobertaConfig.from_pretrained('roberta-base')
    sba_config.is_decoder = True
    sba_config.add_cross_attention = True
    sba_config.num_hidden_layers = args.layers
    model = SBA(sba_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        warmup_steps=args.warmup,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=args.output_dir+'/logs',  # directory for storing logs
        logging_steps=30,
        learning_rate=args.lr,
        save_steps=10000,
        fp16=True
    )

    trainer = SBATrainer(model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                         )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../ragae/msmarco/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result/"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    args = parser.parse_args()

    main(args)