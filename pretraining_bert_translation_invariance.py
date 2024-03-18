
# !pip install transformers datasets
# !pip install magic-timer
# !pip install accelerate -U
# # !pip install transformers[torch]
# !pip install wandb
# !pip install pynvml

# Commented out IPython magic to ensure Python compatibility.
# %env WANDB_DISABLED="true"

import json
from pathlib import Path
from typing import Iterator
import time
import torch.nn as nn
import datasets
import matplotlib.pyplot as plt
import pandas as pd
import pynvml
import torch
from tokenizers import BertWordPieceTokenizer, Regex, normalizers
from tqdm import tqdm
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from library.transformers.src.transformers.models.bert.modeling_bert import BertForMaskedLM
from library.transformers.src.transformers.models.bert.configuration_bert import BertConfig
import os
from os import path

os.environ["WANDB_DISABLED"] = "true"

# Print hardware information
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# gpu_name = pynvml.nvmlDeviceGetName(handle)
# gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)
# print(f"GPU: {gpu_name}, {gpu_mem} GB")
# print(f"{torch.cuda.is_available() = }")

LIMIT_DATASET = 2016  # keep small for development, set to None for full dataset

RANDOM_SEED = 42
NUM_TOKENIZER_TRAINING_ITEMS = 1_000_000  # I made this up, but it seems reasonable
VOCAB_SIZE = 32_768  # from Cramming
DEVICE_BATCH_SIZE = 512 # adjust to get near 100% gpu memory use
MODEL_MAX_SEQ_LEN = 96  # from Cramming
HIDDEN_SIZE = 128

gradient_accumulation_steps = 2048 // DEVICE_BATCH_SIZE  # roughly based on Cramming
batch_size = DEVICE_BATCH_SIZE * gradient_accumulation_steps
print(f"{DEVICE_BATCH_SIZE = }")
print(f"{gradient_accumulation_steps = }")
print(f"{batch_size = }")

RUN_DIR = Path("data") / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
CHECKPOINT_DIR = RUN_DIR / "training_checkpoints"
MODEL_DIR = RUN_DIR / "model"
TOKENIZER_PATH = RUN_DIR / "tokenizer.json"
TRAINER_HISTORY_PATH = RUN_DIR / "trainer_history.json"

RUN_DIR.mkdir(exist_ok=True, parents=True)

from magic_timer import MagicTimer
with MagicTimer() as timer:
    dataset = datasets.load_dataset(
        "wikipedia",
        "20220301.en",
        split=f"train[:{LIMIT_DATASET}]" if LIMIT_DATASET else "train",
        trust_remote_code=True
    )
print(f"Loaded dataset in {timer}")

dataset = dataset.train_test_split(test_size = 0.1, shuffle = True)

tokenizer = BertWordPieceTokenizer()
tokenizer._tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace(Regex("(``|'')"), '"'),
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
        normalizers.Replace(Regex(r"[^\x00-\x7F]+"), ""),
    ]
)  # Normalizer based on, https://github.com/JonasGeiping/cramming/blob/50bd06a65a4cd4a3dd6ee9ecce1809e1a9085374/cramming/data/tokenizer_preparation.py#L52

def tokenizer_training_data(dataset) -> Iterator[str]:
    for i in tqdm(
        range(min(NUM_TOKENIZER_TRAINING_ITEMS, len(dataset))),
        desc="Feeding samples to tokenizer",
    ):
        yield dataset[i]["text"]


with MagicTimer() as timer:
    tokenizer.train_from_iterator(
        tokenizer_training_data(dataset["train"]),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
    )
print(f"Tokenizer trained in {timer}.")
tokenizer.save(str(TOKENIZER_PATH))

model_config = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    max_sequence_length = MODEL_MAX_SEQ_LEN,
    num_hidden_layers = 2,
    num_attention_heads = 2,
    intermediate_size=4*HIDDEN_SIZE,
    max_position_embeddings=HIDDEN_SIZE,

)
model = BertForMaskedLM(model_config, max_sequence_length=96, mlp_layers=[1] )
tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

# class TokenizedDataset(torch.utils.data.Dataset):
#     "This wraps the dataset and tokenizes it, ready for the model"

#     def __init__(self, dataset, tokenizer):
#         self.dataset = dataset
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, i):
#         return self.tokenizer.encode(
#             self.dataset[i]["text"],
#             return_tensors="pt",
#             truncation=True,
#             max_length=MODEL_MAX_SEQ_LEN,
#             padding="max_length",
#             return_special_tokens_mask=True,
#             stride=MODEL_MAX_SEQ_LEN-1
#         )[0, ...]
print(dataset["train"].shape)
def offset_dataset(dataset, tokenizer, max_length, verbose=False, stride = 0):
    def tokenization(example):
        return tokenizer(example["text"],
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length, 
                        padding='max_length',
                        return_special_tokens_mask=True,
                        return_overflowing_tokens=True,
                        stride=stride)
    
    dataset = dataset.map(tokenization, remove_columns=dataset.column_names, batched = True, batch_size = 20)
    dataset = dataset.shuffle(seed=42)
    if verbose:
        print("Preprocessed dataset:", dataset)
    return dataset

tokenized_dataset_train = offset_dataset(dataset["train"], tokenizer,MODEL_MAX_SEQ_LEN, verbose=False, stride= MODEL_MAX_SEQ_LEN - 1)
# tokenized_dataset_train = TokenizedDataset(dataset["train"], tokenizer)
tokenized_dataset_eval = offset_dataset(dataset["test"], tokenizer,MODEL_MAX_SEQ_LEN, verbose=False)
# dataset = datasets.concatenate_datasets(dataset)
print(tokenized_dataset_train.shape)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt",    
)

training_args = TrainingArguments(
    # Optimizer values are from Cramming
    learning_rate=1e-3,
    warmup_ratio=0.5,#2000 steps
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-9,
    weight_decay=0.01,
    max_grad_norm=0.5,
    num_train_epochs=40,
    per_device_train_batch_size=DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=gradient_accumulation_steps,
    dataloader_num_workers=2,
    save_strategy = "no",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=1,
    output_dir=CHECKPOINT_DIR,
    optim="adamw_torch",
    report_to=None,
)
Trainer._get_train_sampler = lambda _: None  # prevent shuffling the dataset again
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_train,
    eval_dataset = tokenized_dataset_eval
)

with MagicTimer() as timer:
    trainer.train()
print(f"Trained model in {timer}.")
# trainer.save_model(str(MODEL_DIR))
TRAINER_HISTORY_PATH.write_text(json.dumps(trainer.state.log_history))

torch.save(model.state_dict(), str(MODEL_DIR))
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

trainer_history = trainer.state.log_history
# trainer_history.loss.plot(label="train loss")
# trainer_history.eval_loss.plot(label = "test loss")
# plt.ylabel("loss")
# plt.savefig(RUN_DIR / "loss.png")
# plt.clf()

trainer_history_train = [item for item in trainer_history if item.get("loss") is not None]
trainer_history_eval = [item for item in trainer_history if item.get("eval_loss") is not None]
trainer_history_train = pd.DataFrame(trainer_history_train[:-1]).set_index("step")
trainer_history_eval = pd.DataFrame(trainer_history_eval[:-1]).set_index("step")

trainer_history_train.loss.plot(label="Train Loss")
trainer_history_eval.eval_loss.plot(label="Test Loss")
plt.legend()
plt.ylabel("loss")
plt.savefig(RUN_DIR / "loss.png")
plt.clf()


def visualizMLPWeights(model):
  """
  visualize the paramters of this layer
  """
  weight_matrix = model.state_dict()["bert.encoder.layer.0.mlp.dense.weight"].detach().cpu()
  plt.imshow(weight_matrix, cmap='hot', interpolation='nearest')
  plt.colorbar()
  plt.savefig(path.join(RUN_DIR, "weights.png"))
  plt.clf()

visualizMLPWeights(model)
