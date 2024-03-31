import json
from pathlib import Path
import torch
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from magic_timer import MagicTimer

from library.transformers.src.transformers.models.bert.modeling_bert import BertForMaskedLM
from library.transformers.src.transformers.models.bert.configuration_bert import BertConfig
import os
from dataset import create_dataset, offset_dataset
from tokenizer import create_tokenizer
from visualization import visualiz_MLPweights, plot_test_train_loss
import argparse

os.environ["WANDB_DISABLED"] = "true"

#Constants
RANDOM_SEED = 42
NUM_TOKENIZER_TRAINING_ITEMS = 1_000_000
VOCAB_SIZE = 32_768 
HIDDEN_SIZE = 128
VERBOSE = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("max_seq_length", type=int, help="Max sequence length to use. Affects the \
                        input/output dimension of the MLP.")

    # training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument('--epoch', default=1, type=int, help="Number of pass through the dataset")

    # datasets
    parser.add_argument("--wiki", action='store_true', help="Whether to use wiki dataset")
    parser.add_argument("--book", action="store_true", help="Whether to use bookcorpus dataset")
    parser.add_argument("--n_examples", type=int, default=None, help="Number of examples from dataset used. \
                        Mainly for specifying a small number for code testing. Default is None, and uses \
                        the entire dataset.")
    parser.add_argumnet('--offset',default = 0, type=int, help = "Offset dataset")
    # parser.add_argument("--test_train_split", type = float, default= 0.1)
    
    # experiment name for saving checkpoints, trained model, config, etc.
    parser.add_argument('--exp_name', default='test', type=str)

    args = parser.parse_args()

    RUN_DIR = Path("experiments") / f"{args.exp_name}"
    MODEL_DIR = RUN_DIR / "model"
    TOKENIZER_PATH = RUN_DIR / "tokenizer.json"
    TRAINER_HISTORY_PATH = RUN_DIR / "trainer_history.json"

    RUN_DIR.mkdir(exist_ok=True, parents=True)

    with MagicTimer() as timer:
        dataset = create_dataset(args.wiki, args.book, args.n_examples, VERBOSE)
        print(f"Loaded dataset in {timer}")
    
    #split data into train and validation sets
    dataset = dataset.train_test_split(test_size = 0.1, shuffle = False)

    #train and load tokenizer
    create_tokenizer(dataset["train"], VOCAB_SIZE, str(TOKENIZER_PATH))
    tokenizer = BertTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))

    tokenized_dataset_train = offset_dataset(dataset["train"], tokenizer,args.max_seq_length, verbose=False, stride= args.max_seq_length - args.offset)
    tokenized_dataset_eval = offset_dataset(dataset["test"], tokenizer, verbose=False)

    #load BERT model
    model_config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        max_sequence_length = args.max_seq_length,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size=4*HIDDEN_SIZE,
        max_position_embeddings=HIDDEN_SIZE,

    )
    model = BertForMaskedLM(model_config, max_sequence_length=args.max_seq_length, mlp_layers=[1])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        return_tensors="pt",
    )

    #trainer
    gradient_accumulation_steps = 2048 // args.batch_size
    batch_size = args.batch_size * gradient_accumulation_steps
    training_args = TrainingArguments(
        learning_rate=args.lr,
        warmup_ratio=0.5,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-9,
        weight_decay=0.01,
        max_grad_norm=0.5,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=2,
        save_strategy = "no",
        # evaluation_strategy="steps",
        # eval_steps=10,
        logging_steps=1,
        optim="adamw_torch",
        report_to=None,
    )
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

    TRAINER_HISTORY_PATH.write_text(json.dumps(trainer.state.log_history))
    torch.save(model.state_dict(), str(MODEL_DIR))

    #plot 
    visualiz_MLPweights(model, RUN_DIR)
    plot_test_train_loss(trainer.state.log_history, RUN_DIR)
