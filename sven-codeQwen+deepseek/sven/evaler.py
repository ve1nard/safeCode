import os
import re
import abc
import torch
import numpy as np

from sven.model import load_model
from sven.constant import PROMPTS
from sven.utils import try_parse

class EvalerBase:
    def __init__(self, args):
        self.args = args
        self.load_model()

    @abc.abstractclassmethod
    def load_model(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def sample(self, file_context, func_context, control, lang):
        raise NotImplementedError()

    def truncate(self, completion, lang):
        if lang == 'py':
            for m in re.finditer(r'\n', completion):
                i, j = m.start(), m.end()
                if j < len(completion) and not completion[j].isspace():
                    return completion[:i]
            if '\n    #' in completion:
                return completion[:completion.rfind('\n    #')]
        else:  # C
            if '\n}' in completion:
                completion = completion[:completion.find('\n}')+2]
            for marker in ('\n    //', '\n    /*'):
                if marker in completion:
                    completion = completion[:completion.rfind(marker)].rstrip() + '\n}'
            lines = [l for l in completion.split('\n') if '->name = "' not in l]
            completion = '\n'.join(lines)
        return completion

    def process_completions(self, prompt_str, prompt_len, gen_output, lang):
        # strip off prompt tokens
        toks = gen_output[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(toks, skip_special_tokens=True)
        outs, ids, dup, bad = [], [], [], []
        for i, comp in enumerate(decoded):
            comp = comp.split(self.tokenizer.eos_token, 1)[0]
            comp = self.truncate(comp, lang)
            comp_ids_len = len(self.tokenizer.encode(comp, add_special_tokens=False))
            full = prompt_str + comp.rstrip() + '\n'
            if full in outs:
                dup.append(full)
            elif try_parse(full, lang) != 0:
                bad.append(full)
            else:
                outs.append(full)
                ids.append((
                    gen_output[i][:prompt_len].tolist(),
                    gen_output[i][prompt_len:prompt_len+comp_ids_len].tolist()
                ))
        return outs, ids, dup, bad

class LMEvaler(EvalerBase):
    def load_model(self):
        self.tokenizer, self.model, self.device = load_model(
            'lm', self.args.model_dir, False, self.args
        )
        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        prompt = file_context + func_context
        if "qwen" in self.args.model_dir.lower():
            messages = [
                {"role": "system",    "content": "You are Qwen, a helpful assistant."},
                {"role": "user",      "content": prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        batch = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        L = batch.input_ids.shape[1]
        gen = self.model.generate(
            **batch,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.process_completions(prompt, L, gen, lang)

class PrefixEvaler(EvalerBase):
    def load_model(self):
        self.tokenizer, self.model, self.device = load_model(
            'prefix', self.args.model_dir, False, self.args
        )
        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        prompt = file_context + func_context
        if "qwen" in self.args.model_dir.lower():
            messages = [
                {"role": "system",    "content": "You are Qwen, a helpful assistant."},
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": ""},  # ready for generation
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        batch = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        L = batch.input_ids.shape[1]
        gen = self.model.generate(
            batch.input_ids,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            control_id=control,
        )
        return self.process_completions(prompt, L, gen, lang)

class TextPromptEvaler(EvalerBase):
    def load_model(self):
        self.tokenizer, self.model, self.device = load_model(
            'lm', self.args.model_dir, False, self.args
        )
        self.model.eval()

    def sample(self, file_context, func_context, control, lang):
        if lang == 'py':
            prompt = file_context + '# ' + PROMPTS[control] + func_context
        else:
            prompt = file_context + '// ' + PROMPTS[control] + func_context
        batch = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        L = batch.input_ids.shape[1]
        gen = self.model.generate(
            **batch,
            do_sample=True,
            num_return_sequences=self.args.num_gen,
            temperature=self.args.temp,
            max_new_tokens=self.args.max_gen_len,
            top_p=self.args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        return self.process_completions(prompt, L, gen, lang)
