# sven/trainer.py
import os, abc
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from sven.model    import load_model, save_model
from sven.dataset  import PrefixDataset, TextPromptDataset
from sven.utils    import set_seed


# ---------- helpers ----------------------------------------------------------
def tokens_to_str(t, tok):
    ids = t.tolist()
    if isinstance(ids[0], list):
        ids = ids[0]
    return tok.decode(ids, skip_special_tokens=False)


# ---------- base class -------------------------------------------------------
class TrainerBase:
    def __init__(self, args):
        self.args = args
        self.tokenizer = self.model = self.device = None

    # ~~~~ to be implemented in subclasses ~~~~
    def load_model   (self): raise NotImplementedError()
    def load_dataset (self): raise NotImplementedError()
    def step         (self, batch): raise NotImplementedError()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ---------- boiler-plate save / log utils ----------
    def _save_ckpt(self, path, step, epoch, opt=None, sch=None):
        os.makedirs(path, exist_ok=True)
        save_model(self.model, path, self.args)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "step.txt"),  "w") as f: f.write(str(step))
        with open(os.path.join(path, "epoch.txt"), "w") as f: f.write(str(epoch))
        if opt: torch.save(opt.state_dict(),  os.path.join(path, "opt.pt"))
        if sch: torch.save(sch.state_dict(),  os.path.join(path, "sch.pt"))

    # ---------- main training loop ----------
    def run(self):
        self.load_model()
        self.load_dataset()

        self.args.logger.info(f"Training args: {self.args}")
        set_seed(self.args)

        train_loader = DataLoader(
            self.dataset,
            sampler=RandomSampler(self.dataset),
            batch_size=1,
            drop_last=True,
        )

        tot_steps = len(self.dataset) // self.args.grad_acc_steps * self.args.num_train_epochs
        opt = AdamW(self.model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        sch = get_linear_schedule_with_warmup(opt, 0, tot_steps)

        self.model.train()
        g_step, acc = 0, OrderedDict()

        for epoch in range(1, self.args.num_train_epochs + 1):
            for step, batch in enumerate(train_loader, 1):
                loss, info = self.step(batch)          # forward/backward handled inside .step
                self._acc(acc, info)

                if step % self.args.logging_steps == 0:
                    self._log(epoch, g_step + step, tot_steps, acc)
                    acc.clear()

            g_step += step

            # checkpoint each epoch
            self._save_ckpt(os.path.join(self.args.output_dir, f"ckpt-{epoch}"),
                            g_step, epoch)

    # --- helpers
    def _acc(self, dic, new):
        for k, v in new.items(): dic[k] = dic.get(k, 0.0) + v

    def _log(self, ep, st, tot, dic):
        msg = ", ".join(f"{k}:{v/ self.args.logging_steps:.4f}" for k, v in dic.items())
        self.args.logger.info(f"Epoch {ep}, step {st}/{tot}: {msg}")


# ============================================================================ #
#                               PREFIX TRAINER                                 #
# ============================================================================ #
class PrefixTrainer(TrainerBase):
    def load_model(self):
        self.tokenizer, self.model, self.device = load_model(
            "prefix", self.args.pretrain_dir, True, self.args
        )
        for n, p in self.model.named_parameters():
            p.requires_grad = n.startswith("prefix_params")
        self.model.train()

    def load_dataset(self):
        self.dataset     = PrefixDataset(self.args, self.tokenizer, "train")
        self.val_dataset = PrefixDataset(self.args, self.tokenizer, "val")

    # -------------- core training step --------------
    def step(self, batch):
        """
        Supports *two* batch formats:
          ( ids, labels )                           # chat-style Qwen
          ( tokens, weights, ctrl_id, vul_id )      # regular prefix-tuning
        """
        if len(batch) == 2:                                        # ← chat path
            ids, labels = (t.to(self.device) for t in batch)
            out = self.model(input_ids=ids,
                              labels=labels,
                              use_cache=False)
            loss = out.loss
            loss.backward()
            return loss, {"loss": loss.item()}

        # ---------- classic prefix-tuning path ----------
        tokens, weights, ctrl_ids, _ = batch
        tokens   = tokens.to(self.device).unsqueeze(0)   # [1,T]
        ctrl_id  = int(ctrl_ids.item())

        # 1) encode user part to build cache
        asst_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
        split   = (tokens[0] == asst_id).nonzero(as_tuple=False)[0,0] + 1
        user, assist = tokens[:, :split], tokens[:, split:]

        with torch.no_grad():                       # cache build
            past = self.model(input_ids=user,
                               control_id=ctrl_id,
                               use_cache=True).past_key_values

        out = self.model(input_ids      = assist,
                         past_key_values = past,
                         labels         = assist,
                         use_cache      = False)
        loss = out.loss
        loss.backward()
        return loss, {"loss": loss.item()}


# ============================================================================ #
#                            TEXT-PROMPT TRAINER                               #
# ============================================================================ #
class TextPromptTrainer(TrainerBase):
    def load_model(self):
        self.tokenizer, self.model, self.device = load_model(
            "lm", self.args.pretrain_dir, True, self.args
        )
        self.model.train()

    def load_dataset(self):
        self.dataset     = TextPromptDataset(self.args, self.tokenizer, "train")
        self.val_dataset = TextPromptDataset(self.args, self.tokenizer, "val")

    def step(self, batch):
        ids, labels = (t.to(self.device) for t in batch)
        out  = self.model(ids, labels=labels, use_cache=False)
        out.loss.backward()
        return out.loss, {"loss": out.loss.item()}
