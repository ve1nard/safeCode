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

import tempfile
import shutil
import csv
import json
import os
import torch
from collections import OrderedDict

def eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario):
    s_in_dir = os.path.join(data_dir, scenario)
    with open(os.path.join(s_in_dir, 'info.json')) as f:
        info = json.load(f)
    with open(os.path.join(s_in_dir, 'file_context.' + info['language'])) as f:
        file_context = f.read()
    with open(os.path.join(s_in_dir, 'func_context.' + info['language'])) as f:
        func_context = f.read()

    for control_id, control in enumerate(controls):
        total = 0
        vul = 0
        sec = 0
        success = False
        attempts = 0

        while attempts < args.max_attempts:
            set_seed(args)
            with torch.no_grad():
                outputs, output_ids, _, _ = evaler.sample(file_context, func_context, control_id, info['language'])

            for i, output in enumerate(outputs):
                total += 1

                with tempfile.TemporaryDirectory() as temp_dir:
                    fname = f'gen.{info["language"]}'
                    src_path = os.path.join(temp_dir, fname)
                    with open(src_path, 'w') as f:
                        f.write(output)

                    db_path = os.path.join(temp_dir, 'codeql_db')
                    csv_path = os.path.join(temp_dir, 'codeql_out.csv')

                    codeql_create_db(info, temp_dir, db_path)
                    codeql_analyze(info, db_path, csv_path)

                    if vul_type == 'cwe-078':
                        filter_cwe78_fps(temp_dir, control)

                    # Check for vulnerabilities in CSV
                    is_vulnerable = False
                    with open(csv_path) as csv_f:
                        reader = csv.reader(csv_f)
                        for row in reader:
                            if len(row) >= 5:
                                is_vulnerable = True
                                break

                    if is_vulnerable:
                        vul += 1
                    else:
                        sec += 1
                        success = True
                        break  # Stop after first secure output

            if success:
                break
            attempts += 1

        yield {
            "vul_type": vul_type,
            "scenario": scenario,
            "control": control,
            "attempts": attempts + 1,
            "success": success,
            "total": total,
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