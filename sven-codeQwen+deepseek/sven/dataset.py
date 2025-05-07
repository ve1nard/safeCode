import os
import abc
import json
import torch
from torch.utils.data import Dataset

from sven.constant import (
    BINARY_LABELS,
    SEC_LABEL,
    VUL_LABEL,
    PROMPTS,
    CWES_TRAINED,
    CWES_TRAINED_SUBSET,
)
from sven.utils import get_indent


def format_as_chat(example: dict, tokenizer) -> str:
    """
    Uses tokenizer's chat template to format the conversation.
    Only for Qwen-style models.
    """
    messages = [
        {"role": "user",      "content": example.get("func_src_before", "").strip()},
        {"role": "assistant", "content": example.get("func_src_after",  "").strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


class DatasetBase(Dataset):
    def __init__(self, args, tokenizer, mode):
        self.args      = args
        self.tokenizer = tokenizer
        self.dataset   = []

        # pick your CWE types
        if self.args.vul_type:
            vul_types = [self.args.vul_type]
        else:
            vul_types = (
                CWES_TRAINED_SUBSET
                if "incoder" in self.args.pretrain_dir
                else CWES_TRAINED
            )

        for i, cwe in enumerate(vul_types):
            path = os.path.join(args.data_dir, mode, f"{cwe}.jsonl")
            with open(path) as f:
                for line in f:
                    diff_j = json.loads(line)
                    lang = "py" if diff_j["file_name"].endswith(".py") else "c"
                    labels = [SEC_LABEL, VUL_LABEL]
                    srcs   = [diff_j["func_src_after"], diff_j["func_src_before"]]
                    # pick diff granularity
                    if   args.diff_level == "prog": diffs = [None, None]
                    elif args.diff_level == "line":
                        diffs = [diff_j["line_changes"]["added"],
                                 diff_j["line_changes"]["deleted"]]
                    elif args.diff_level == "char":
                        diffs = [diff_j["char_changes"]["added"],
                                 diff_j["char_changes"]["deleted"]]
                    elif args.diff_level == "mix":
                        diffs = [diff_j["char_changes"]["added"],
                                 diff_j["line_changes"]["deleted"]]
                    else:
                        raise NotImplementedError()

                    for label, src, changes in zip(labels, srcs, diffs):
                        self.add_data(label, src, changes, i, lang)

    @abc.abstractclassmethod
    def add_data(self, label, src, changes, vul_id, lang):
        raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        toks, labs = self.dataset[idx]
        return torch.tensor(toks, dtype=torch.long), torch.tensor(labs, dtype=torch.long)


class PrefixDataset(DatasetBase):
    def __init__(self, args, tokenizer, mode):
        super().__init__(args, tokenizer, mode)

    def add_data(self, label, src, changes, vul_id, lang):
        # Qwen chat-style fine-tuning
        if "qwen" in self.args.pretrain_dir.lower():
            example = {"func_src_before": src, "func_src_after": src}
            chat = format_as_chat(example, self.tokenizer)
            # truncate to max tokens
            enc = self.tokenizer(
                chat,
                truncation=True,
                max_length=self.args.max_num_tokens,
                return_tensors="pt"
            )
            input_ids = enc.input_ids[0].tolist()
            if len(input_ids) < 2:
                return
            # build labels: only after assistant_start
            asst_id = self.tokenizer.convert_tokens_to_ids("<|assistant|>")
            seen = False
            labels = []
            for t in input_ids:
                if t == asst_id:
                    seen = True
                labels.append(t if seen else -100)
            self.dataset.append((input_ids, labels))
            return

        # standard prefix-tuning path
        control_id = BINARY_LABELS.index(label)
        data = self.get_tensor(src, vul_id, control_id, changes)
        if data is not None:
            self.dataset.append(data)

    def get_tensor(self, src, vul_id, control_id, changes):
        be = self.tokenizer.encode_plus(src)
        tokens = be.data["input_ids"]
        if len(tokens) > self.args.max_num_tokens:
            return None
        # mask for changed spans
        if changes is None:
            weights = [1] * len(tokens)
        else:
            weights = [0] * len(tokens)
            for c in changes:
                i0 = be.char_to_token(c["char_start"])
                i1 = be.char_to_token(c["char_end"] - 1)
                for i in range(i0, i1+1):
                    weights[i] = 1
            min_tok = 2 if self.args.vul_type in ("cwe-invalid","cwe-valid") else 1
            if sum(weights) < min_tok or len(tokens)-sum(weights) < min_tok:
                return None
        return tokens, weights, control_id, vul_id


class TextPromptDataset(DatasetBase):
    def __init__(self, args, tokenizer, mode):
        super().__init__(args, tokenizer, mode)

    def add_data(self, label, src, changes, vul_id, lang):
        control = PROMPTS[BINARY_LABELS.index(label)]
        if lang == "py":
            prompt = get_indent(src) + "# " + control
        else:
            prompt = get_indent(src) + "// " + control
        src2 = prompt + src
        data = self.get_tensor(src2, control, changes)
        if data is not None:
            self.dataset.append(data)

    def get_tensor(self, src, control, changes):
        be = self.tokenizer.encode_plus(src)
        tokens = be.data["input_ids"]
        if changes is None:
            labels = tokens[:]
        else:
            labels = [-100]*len(tokens)
            ok = False
            for c in changes:
                s = c["char_start"] + len(control)
                e = c["char_end"]   + len(control)
                i0 = be.char_to_token(s)
                i1 = be.char_to_token(e-1)
                for i in range(i0, i1+1):
                    labels[i] = tokens[i]
                    ok = True
            if not ok:
                return None
        if len(tokens) > self.args.max_num_tokens:
            return None
        return tokens, labels
