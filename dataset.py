from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast

def create_dataset(wiki, book, n_examples=None, verbose=False):
    """
    wiki: (bool) whether to use wiki dataset
    book: (bool) whether to use bookcorpus dataset
    n_examples: (int) number of examples from dataset to use
    """
    assert wiki or book       # at least use one dataset
    if verbose:
        print(f"Using dataset(s): {'wiki ' if wiki else ''}{'bookcorpus ' if book else ''}")

    n_datasets = int(wiki) + int(book)
    if n_examples is not None:
        n_examples_per_dataset = n_examples // n_datasets
    else:
        n_examples_per_dataset = ""

    combined_dataset = []
    if wiki:
        wiki = load_dataset("wikipedia", "20220301.en", split=f'train[:{n_examples_per_dataset}]', trust_remote_code=True)
        combined_dataset.append(wiki)
    if book:
        book = load_dataset("bookcorpus", split=f'train[:{n_examples_per_dataset}]')
        combined_dataset.append(book)

    combined_dataset = concatenate_datasets(combined_dataset)
    if verbose:
        print(combined_dataset)

    # test:
    # Command; n_rows
    # 1. --wiki; 6458670
    # 2. --book; 74004228
    # 3. --wiki --book; 80462898
    # 4. --wiki --n_examples 10000; 10000
    # 5. --book --n_examples 10000; 10000
    # 6. --wiki --book --n_examples 10000; 10000

    return combined_dataset

def preprocess_for_distillation(dataset, bert_model, max_length, verbose=False):
    """
    Preprocess dataset for distillation: 
    1. truncate long sequence to fixed length
    2. pad short sequence to fixed length 

    bert_model: the BERT model to use
    max_length: the fixed sequence length
    """

    tokenizer = BertTokenizerFast.from_pretrained(bert_model)

    def tokenization(example):
        # return_overflowing_tokens: break a long sequence into chunks of max_length
        # can set stride=n to perform sliding window.
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_length, return_overflowing_tokens=True)

    dataset = dataset.map(tokenization, batched=True, remove_columns=dataset.column_names)

    if verbose:
        print("Preprocessed dataset:", dataset)

    return dataset

def preprocess_for_pretraining_bert(dataset, bert_model, max_length = 512, verbose = False):
    """"
    Preprocess dataset for pretraining bert:
    1. truncate long sequence to fixed length
    2. pad short sequence to fixed length 

    bert_model: the BERT model to use
    max_length: the fixed sequence length
    """
    preprocess_for_distillation(dataset, bert_model, max_length, verbose)

    # data_collator = DataCollatorForLanguageModeling(
    # tokenizer=tokenizer,
    # mlm=True,
    # mlm_probability=0.15,
    # return_tensors="pt",
# )
    

