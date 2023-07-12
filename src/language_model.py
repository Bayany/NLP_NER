
import logging
import sys
targets = logging.StreamHandler(sys.stdout), logging.FileHandler('logs/language_model.log')
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=targets)


import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from glob import glob


torch.manual_seed(42)


class WebtoonDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]



def prepare_data(label):

    # corpus = []
    # if(label=="all_chapters"):
    #     data_path="data/clean/PurpleHyacinth/*.txt"
    #     files = glob(data_path)

    #     for file_path in files:
    #         with open(file_path,encoding="utf-8") as f:
    #             text=f.read()
    #             corpus.extend([text[i: text[i:i+400].rindex(" ")] for i in range(0, len(text), 400) ])
    # else:
    label_df = pd.read_csv(f"data/sentencebroken/{label}.csv")
    corpus = [row["sentence"] for i,row in label_df.iterrows()]
    return corpus



if __name__=="__main__":

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').cuda()
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(output_dir='./models', num_train_epochs=10, logging_steps=100, save_steps=5000,
                                      per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                      warmup_steps=10, weight_decay=0.05, logging_dir='./logs', report_to = 'none')
    
    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                                  'labels': torch.stack([f[0] for f in data])}


    for label  in ["PERSON","LOC", "all_chapters"]:
        logging.info(f"Training language_model ({label})")

        corpus= prepare_data(label)
        max_length = max([len(tokenizer.encode(text)) for text in corpus])
        dataset = WebtoonDataset(corpus, tokenizer, max_length=max_length)
        train_size = int(0.9 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        torch.cuda.empty_cache()
        Trainer(model=model,  args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=data_collator).train()

        generated = tokenizer("<|startoftext|> ", return_tensors="pt").input_ids.cuda()
        sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                        max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
        torch.save(model.state_dict(), f'models/{label}.language_model.pt')
        with open(f"stats/{label}.language_model.txt","w",encoding="utf-8") as f:
            for i, sample_output in enumerate(sample_outputs):
                f.write("{}- {}\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
