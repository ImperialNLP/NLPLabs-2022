import pytorch_lightning as pl
import torch
import torch.nn as nn
from layers.transformers import Transformer
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.utils.data import DataLoader
import math as m

class TransformerTrainer(pl.LightningModule):
    def __init__(self, src_vocab: Vocab, trg_vocab: Vocab, warmup_steps=4000, d_model=256, d_ff=1024, num_layers=6, num_heads=8, device="cpu", dropout=0.3):
        super().__init__()
    
        self.model = Transformer(
            src_vocab_len=len(src_vocab),
            trg_vocab_len=len(trg_vocab),
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            src_pad_idx=src_vocab.__getitem__("<pad>"),
            trg_pad_idx=trg_vocab.__getitem__("<pad>"),
            device=device
        )
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device_ = device
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        self.criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.__getitem__("<pad>"))

    def training_step(self, batch, batch_idx):
        src = batch[0].to(self.device_)
        trg = batch[1].to(self.device_)
        trg_input = trg[:, :-1]

        logits = self.model(src, trg_input)

        ys = trg[:, 1:]
        loss = self.criterion(logits.permute(0, 2, 1), ys)

        self.change_lr_in_optimizer()

        return loss

    def validation_step(self, batch, batch_idx):
        src = batch[0].to(self.device_)
        trg = batch[1].to(self.device_)
        trg_input = trg[:, :-1]

        logits = self.model(src, trg_input)

        ys = trg[:, 1:]
        val_loss = self.criterion(logits.permute(0, 2, 1), ys)
        self.log("val loss", val_loss)

        for idx in range(len(src)):
            print(" SRC:\t", self.clean_and_print_tokens(src[idx], "src"))
            print(" TRG:\t", self.clean_and_print_tokens(trg[idx], "trg"))
            print("PRED:\t", self.clean_and_print_tokens(torch.argmax(logits[idx], dim=-1), "trg"))
            print("")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def change_lr_in_optimizer(self):
        min_arg1 = m.sqrt(1/(self.global_step+1))
        min_arg2 = self.global_step * (self.warmup_steps**-1.5)
        lr = m.sqrt(1/self.d_model) * min(min_arg1, min_arg2)
        self.trainer.lightning_optimizers[0].param_groups[0]['lr'] = lr

    def clean_and_print_tokens(self, tokens, src_or_trg):
        if src_or_trg == "src":
            vocab = self.src_vocab
        elif src_or_trg == "trg":
            vocab = self.trg_vocab

        return " ".join(vocab.lookup_tokens(tokens.tolist()))

if __name__ == "__main__":
    device = ("cuda:0" if torch.cuda.is_available else "cpu")
    train_iter, val_iter, test_iter = Multi30k()
    src_tokenizer = get_tokenizer("basic_english")
    trg_tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter, src_or_trg):
        for batch in data_iter:
            if src_or_trg == "src":
                yield src_tokenizer(batch[0])
            elif src_or_trg == "trg":
                yield trg_tokenizer(batch[1])


    src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, "src"), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
    src_vocab.set_default_index(src_vocab["<unk>"])
    train_iter, val_iter, test_iter = Multi30k()

    trg_vocab = build_vocab_from_iterator(yield_tokens(train_iter, "trg"), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
    trg_vocab.set_default_index(trg_vocab["<unk>"])
    train_iter, val_iter, test_iter = Multi30k()

    MAX_SEQ_LEN = 30
    def pad_to_max(tokens):
        return tokens[:MAX_SEQ_LEN] + ["<pad>"] * max(0, MAX_SEQ_LEN - len(tokens))

    def collate_fn(batch):
        srcs = []
        trgs = []
        for pair in batch:
            src = pair[0]
            trg = pair[1]

            tokenized_src = src_vocab(pad_to_max(src_tokenizer("<sos> " + src + " <eos>")))
            tokenized_trg = trg_vocab(pad_to_max(trg_tokenizer("<sos> " + trg + " <eos>")))

            srcs.append(tokenized_src)
            trgs.append(tokenized_trg)

        srcs = torch.tensor(srcs, dtype=torch.long)
        trgs = torch.tensor(trgs, dtype=torch.long)
        return srcs, trgs

    dataloader = DataLoader(list(train_iter), batch_size=64, shuffle=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(list(val_iter), batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(list(test_iter), batch_size=64, shuffle=False, collate_fn=collate_fn)

    transformer = TransformerTrainer(src_vocab, trg_vocab, device=device)
    trainer = pl.Trainer(gpus=1, min_epochs=20)
    trainer.fit(transformer, dataloader, val_dataloader)