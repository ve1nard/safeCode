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

# SecCoder implementation begin

from glob import glob
from InstructorEmbedding import INSTRUCTOR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# SecCoder implementation end

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained_subset', 'prompts', 'gen_1', 'gen_2'], default='trained')
    parser.add_argument('--vul_type', type=str, default=None)
    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix', 'text'], default='prefix')
    parser.add_argument('--model_dir', type=str, default=None)

    parser.add_argument('--data_dir', type=str, default='../data_eval')
    parser.add_argument('--output_dir', type=str, default='../experiments/sec_eval')

    parser.add_argument('--num_gen', type=int, default=25)
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

def eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario, demonstration_set):
    s_out_dir = os.path.join(output_dir, scenario)
    os.makedirs(s_out_dir)
    s_in_dir = os.path.join(data_dir, scenario)
    with open(os.path.join(s_in_dir, 'info.json')) as f:
        info = json.load(f)
    with open(os.path.join(s_in_dir, 'file_context.'+info['language'])) as f:
        file_context = f.read()
    with open(os.path.join(s_in_dir, 'func_context.'+info['language'])) as f:
        func_context = f.read()

    # SecCoder implementation start

    # Having the prompt (which is a combination of file_context and func_context)
    # and the demonstration set in hand, the best matching secure code example is
    # found using the Instructor embedding model. 
    prompt = file_context + func_context
    # The large model is used to follow the paper.
    model = INSTRUCTOR('hkunlp/instructor-large')
    # A custom instruction is used to create better embeddings for the demonstration
    # set and the prompt 
    task_objective_prompt = 'Represent the code for retrieving similar code snippets:'
    task_objective_demonstration = 'Represent the code snippet used as an example of secure code that avoids a particular vulnerability'

    demonstration_set_instruction = [[task_objective_demonstration, snippet] for snippet in demonstration_set]
    demonstration_set_embedding = model.encode(demonstration_set_instruction)

   # task_objective_prompt = 'Represent the code snippet used as a prompt for code generation LLM for completion'
    prompt_embedding = model.encode([[task_objective_prompt, prompt]])

    # Find the best matching complete secure code example for a given prompt
    similarities = cosine_similarity(prompt_embedding, demonstration_set_embedding)
    best_match_index = np.argmax(similarities)
    

    # Insert the selected code snippet at the beginning of the prompt before any imports.
    # To do this, the code is inserted before file_context using the template specified in
    # the paper.
    if info['language'] == 'py':
        formatted_snippet = f'\"\"\"\n```\n{demonstration_set[best_match_index]}\n```\n\"\"\"\n'
    elif info['language'] == 'c':
        formatted_snippet = f'#if 0\n```\n{demonstration_set[best_match_index]}\n```\n#endif\n'
    else:
        raise NotImplementedError("ONLY PYTHON AND C AT THE MOMENT")
    #formatted_snippet = f"""```\ndo good\n```"""
    file_context = formatted_snippet + file_context

    # SecCoder implementation end

    for control_id, control in enumerate(controls):
        set_seed(args)
        with torch.no_grad():
            outputs, output_ids, dup_srcs, non_parsed_srcs = evaler.sample(file_context, func_context, control_id, info['language'])

        out_src_dir = os.path.join(s_out_dir, f'{control}_output')
        os.makedirs(out_src_dir)
        output_ids_j = OrderedDict()
        all_fnames = set()
        for i, (output, output_id) in enumerate(zip(outputs, output_ids)):
            fname = f'{str(i).zfill(2)}.'+info['language']
            all_fnames.add(fname)
            with open(os.path.join(out_src_dir, fname), 'w') as f:
                f.write(output)
            output_ids_j[fname] = output_id
        with open(os.path.join(s_out_dir, f'{control}_output_ids.json'), 'w') as f:
            json.dump(output_ids_j, f, indent=2)
        if info['language'] == 'c':
            shutil.copy2('Makefile', out_src_dir)

        for srcs, name in [(dup_srcs, 'dup'), (non_parsed_srcs, 'non_parsed')]:
            src_dir = os.path.join(s_out_dir, f'{control}_{name}')
            os.makedirs(src_dir)
            for i, src in enumerate(srcs):
                fname = f'{str(i).zfill(2)}.'+info['language']
                with open(os.path.join(src_dir, fname), 'w') as f:
                    f.write(src)

        vuls = set()
        if len(outputs) != 0:
            csv_path = os.path.join(s_out_dir, f'{control}_codeql.csv')
            db_path = os.path.join(s_out_dir, f'{control}_codeql_db')
            codeql_create_db(info, out_src_dir, db_path)
            codeql_analyze(info, db_path, csv_path)
            if vul_type == 'cwe-078':
                filter_cwe78_fps(s_out_dir, control)
            with open(csv_path) as csv_f:
                reader = csv.reader(csv_f)
                for row in reader:
                    if len(row) < 5: continue
                    out_src_fname = row[-5].replace('/', '')
                    vuls.add(out_src_fname)
        secs = all_fnames - vuls

        d = OrderedDict()
        d['vul_type'] = vul_type
        d['scenario'] = scenario
        d['control'] = control
        d['total'] = len(all_fnames)
        d['sec'] = len(secs)
        d['vul'] = len(vuls)
        d['dup'] = len(dup_srcs)
        d['non_parsed'] = len(non_parsed_srcs)
        d['model_type'] = args.model_type
        d['model_dir'] = args.model_dir
        d['temp'] = args.temp

        yield d

def eval_vul(args, evaler, controls, vul_types):
    for vul_type in vul_types:
        data_dir = os.path.join(args.data_dir, vul_type)
        output_dir = os.path.join(args.output_dir, vul_type)
        os.makedirs(output_dir)

        # SecCoder implementation begin

        # Create a demonstration set by parsing the SVEN training dataset and extracting only 
        # secure implementations.
        cwe_examples_folder = "../data_train_val/train"
        demonstration_set = []
    
        # Get all JSON files in the folder
        json_files = glob(os.path.join(cwe_examples_folder, "*.jsonl"))
        
        # There are 9 CWE categories, and from each cateogry 10 examples are included in the 
        # demonstration dataset. The number of included examples can be experimented with.
        max_per_file = 10
        for file_path in json_files:
            with open(file_path, 'r') as f:
                num_examples = 0
                # Process each line (function entry) in the JSON file
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Fields like 'func_name', 'func_src_before', etc. are not needed
                        # since we are only interested in including complete secure examples
                        secure_code = data["func_src_after"]
                        #print(data, secure_code)
                        demonstration_set.append(secure_code)
                        num_examples += 1
                        # Stop when we have enough from this file
                        if num_examples >= max_per_file:
                            break
                            
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Skipping invalid entry in {file_path}: {e}")
                        continue

        # SecCoder implementation end

        with open(os.path.join(output_dir, 'result.jsonl'), 'w') as f:
            for scenario in list(sorted(os.listdir(data_dir))):
                # SecCoder implementation start

                # The signaature of eval_single is changed to allow passing the demonstration set
                # as a parameter.
                for d in eval_single(args, evaler, controls, output_dir, data_dir, vul_type, scenario, demonstration_set):

                # SecCoder implementation end
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
