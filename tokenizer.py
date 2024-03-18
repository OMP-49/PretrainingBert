from tokenizers import BertWordPieceTokenizer, Regex, normalizers
from tqdm import tqdm
from magic_timer import MagicTimer

# VOCAB_SIZE = 32_768

def create_tokenizer(dataset, vocab_size, tokenizer_path:str):
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
    with MagicTimer() as timer:
        tokenizer.train_from_iterator(
            tokenizer_training_data(dataset["train"]),
            vocab_size=vocab_size,
            min_frequency=2,
        )
    print(f"Tokenizer trained in {timer}.")
    tokenizer.save(tokenizer_path)


def tokenizer_training_data(dataset):
    for i in tqdm(
        range(len(dataset)),
        desc="Feeding samples to tokenizer",
    ):
        yield dataset[i]["text"]


