import os
import csv
import json
import torch
import shutil
import argparse
import subprocess
import libcst as cst
from libcst.metadata import PositionProvider
from libcst._position import CodePosition
from collections import OrderedDict

from sven.evaler import LMEvaler, PrefixEvaler, TextPromptEvaler
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import BINARY_LABELS, MODEL_DIRS, CWES_DICT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained_subset', 'prompts', 'gen_1', 'gen_2'], default='trained')
    parser.add_argument('--vul_type', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix', 'text'], default='prefix')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--max_attempts', type=int, default=5)

    parser.add_argument('--data_dir', type=str, default='../data_eval')
    parser.add_argument('--output_dir', type=str, default='../experiments/sec_eval')

    parser.add_argument('--num_gen', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--max_gen_len', type=int, default=300)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.model_type == 'lm':
        if args.model_dir is None:
            args.model_dir = '2b'
        if args.model_dir in MODEL_DIRS:
            args.model_dir = MODEL_DIRS[args.model_dir]

    args.output_dir = os.path.join(args.output_dir, args.output_name, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)

    return args

def get_evaler(args):
    if args.model_type == 'lm':
        evaler = LMEvaler(args)
        controls = ['orig']
    elif args.model_type == 'prefix':
        evaler = PrefixEvaler(args)
        controls = BINARY_LABELS
    elif args.model_type == 'text':
        evaler = TextPromptEvaler(args)
        controls = BINARY_LABELS
    else:
        raise NotImplementedError()

    return evaler, controls

def codeql_create_db(info, out_src_dir, out_db_dir):
    if info['language'] == 'py':
        cmd = '../codeql/codeql database create {} --quiet --language=python --overwrite --source-root {}'
        cmd = cmd.format(out_db_dir, out_src_dir)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    elif info['language'] == 'c':
        cmd = '../codeql/codeql database create {} --quiet --language=cpp --overwrite --command="make -B" --source-root {}'
        cmd = cmd.format(out_db_dir, out_src_dir)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        raise NotImplementedError()

def codeql_analyze(info, out_db_dir, out_csv_path):
    if info['language'] == 'py':
        cmd = '../codeql/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}'
        cmd = cmd.format(out_db_dir, info['check_ql'], out_csv_path, os.path.expanduser('~/.codeql/packages/codeql/'))
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    elif info['language'] == 'c':
        cmd = '../codeql/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}'
        cmd = cmd.format(out_db_dir, info['check_ql'], out_csv_path, os.path.expanduser('~/.codeql/packages/codeql/'))
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)
    else:
        raise NotImplementedError()

class CWE78Visitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, src, start, end):
        self.list_vars = set()
        self.src = src
        self.start = start
        self.end = end
        self.fp = False

    def visit_Assign(self, node):
        if len(node.targets) != 1: return
        if not isinstance(node.targets[0].target, cst.Name): return
        target_name = node.targets[0].target.value
        if isinstance(node.value, cst.List):
            if len(node.value.elements) == 0: return
            if not isinstance(node.value.elements[0].value, cst.BaseString): return
            self.list_vars.add(target_name)
        elif isinstance(node.value, cst.Name):
            if node.value.value in self.list_vars:
                self.list_vars.add(target_name)
        elif isinstance(node.value, cst.BinaryOperation):
            if isinstance(node.value.left, cst.List):
                self.list_vars.add(target_name)
            elif isinstance(node.value.left, cst.Name) and node.value.left.value in self.list_vars:
                self.list_vars.add(target_name)
            if isinstance(node.value.right, cst.List):
                self.list_vars.add(target_name)
            elif isinstance(node.value.right, cst.Name) and node.value.right.value in self.list_vars:
                self.list_vars.add(target_name)

    def visit_Name(self, node):
        pos = self.get_metadata(PositionProvider, node)
        if self.start.line != pos.start.line: return
        if self.start.column != pos.start.column: return
        if self.end.line != pos.end.line: return
        if self.end.column != pos.end.column: return
        assert pos.start.line == pos.end.line
        if node.value in self.list_vars:
            self.fp = True

def filter_cwe78_fps(s_out_dir, control):
    csv_path = os.path.join(s_out_dir, f'{control}_codeql.csv')
    out_src_dir = os.path.join(s_out_dir, f'{control}_output')
    with open(csv_path) as csv_f:
        lines = csv_f.readlines()
    shutil.copy2(csv_path, csv_path+'.fp')
    with open(csv_path, 'w') as csv_f:
        for line in lines:
            row = line.strip().split(',')
            if len(row) < 5: continue
            out_src_fname = row[-5].replace('/', '').strip('"')
            out_src_path = os.path.join(out_src_dir, out_src_fname)
            with open(out_src_path) as f:
                src = f.read()
            start = CodePosition(int(row[-4].strip('"')), int(row[-3].strip('"'))-1)
            end = CodePosition(int(row[-2].strip('"')), int(row[-1].strip('"')))
            visitor = CWE78Visitor(src, start, end)
            tree = cst.parse_module(src)
            wrapper = cst.MetadataWrapper(tree)
            wrapper.visit(visitor)
            if not visitor.fp:
                csv_f.write(line)

def eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario):
    s_in_dir = os.path.join(data_dir, scenario)
    with open(os.path.join(s_in_dir, 'info.json')) as f:
        info = json.load(f)
    with open(os.path.join(s_in_dir, 'file_context.' + info['language'])) as f:
        file_context = f.read()
    with open(os.path.join(s_in_dir, 'func_context.' + info['language'])) as f:
        func_context = f.read()

    for control_id, control in enumerate(controls):
        s_out_dir = os.path.join(output_dir, scenario)
        os.makedirs(s_out_dir, exist_ok=True)

        out_src_dir = os.path.join(s_out_dir, f'{control}_output')
        non_parsed_dir = os.path.join(s_out_dir, f'{control}_non_parsed')
        dup_dir = os.path.join(s_out_dir, f'{control}_dup')
        os.makedirs(out_src_dir, exist_ok=True)
        os.makedirs(non_parsed_dir, exist_ok=True)
        os.makedirs(dup_dir, exist_ok=True)

        total = 0
        parsed = 0
        non_parsed = 0
        dup = 0
        vul = 0
        sec = 0
        success = False
        attempts = 0
        num_tokens = 0
        output_ids_j = OrderedDict()

        while attempts < args.max_attempts:
            set_seed(args)
            with torch.no_grad():
                outputs, output_ids, dup_srcs, non_parsed_srcs, new_tokens = evaler.sample(
                    file_context, func_context, control_id, info['language']
                )
                num_tokens += new_tokens
            for i, (output, output_id) in enumerate(zip(outputs, output_ids)):
                fname = f'{str(total).zfill(2)}.{info["language"]}'
                total += 1

                # Classify output
                if output in non_parsed_srcs:
                    non_parsed += 1
                    with open(os.path.join(non_parsed_dir, fname), 'w') as f:
                        f.write(output)
                    continue
                elif output in dup_srcs:
                    dup += 1
                    with open(os.path.join(dup_dir, fname), 'w') as f:
                        f.write(output)
                    continue

                parsed += 1
                output_ids_j[fname] = output_id
                out_path = os.path.join(out_src_dir, fname)
                with open(out_path, 'w') as f:
                    f.write(output)

                # Optional for C code
                if info['language'] == 'c':
                    shutil.copy2('Makefile', out_src_dir)

                # Analyze with CodeQL
                db_path = os.path.join(s_out_dir, f'{control}_codeql_db')
                csv_path = os.path.join(s_out_dir, f'{control}_codeql.csv')

                codeql_create_db(info, out_src_dir, db_path)
                codeql_analyze(info, db_path, csv_path)

                if vul_type == 'cwe-078':
                    filter_cwe78_fps(s_out_dir, control)

                is_vulnerable = False
                if os.path.exists(csv_path):
                    with open(csv_path) as csv_f:
                        reader = csv.reader(csv_f)
                        for row in reader:
                            if len(row) >= 5:
                                out_src_fname = row[-5].replace('/', '')
                                if out_src_fname == fname:
                                    is_vulnerable = True
                                    break

                if is_vulnerable:
                    vul += 1
                else:
                    sec += 1
                    success = True
                    break  # Found first secure output → stop here

            if success:
                break
            attempts += 1

        # Save output IDs
        with open(os.path.join(s_out_dir, f'{control}_output_ids.json'), 'w') as f:
            json.dump(output_ids_j, f, indent=2)

        yield {
            "vul_type": vul_type,
            "scenario": scenario,
            "control": control,
            "attempts": attempts + 1,
            "success": success,
            "total": total,
            "tokens": num_tokens,
            "parsed": parsed,
            "non_parsed": non_parsed,
            "dup": dup,
            "vul": vul,
            "sec": sec,
            "model_type": args.model_type,
            "model_dir": args.model_dir,
            "temp": args.temp
        }


def eval_vul(args, evaler, controls, vul_types):
    for vul_type in vul_types:
        data_dir = os.path.join(args.data_dir, vul_type)
        output_dir = os.path.join(args.output_dir, vul_type)
        os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'result.jsonl'), 'w') as f:
            for scenario in list(sorted(os.listdir(data_dir))):
                for d in eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario):
                    s = json.dumps(d)
                    args.logger.info(s)
                    f.write(s+'\n')

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_devices(args)
    set_seed(args)
    args.logger.info(f'args: {args}')

    evaler, controls = get_evaler(args)
    assert args.eval_type in CWES_DICT
    if args.vul_type is not None:
        vul_types = [args.vul_type]
    else:
        vul_types = CWES_DICT[args.eval_type]

    eval_vul(args, evaler, controls, vul_types)

if __name__ == '__main__':
    main()
