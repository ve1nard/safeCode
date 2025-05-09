import os
import csv
import json
import shutil
import argparse
import subprocess
import libcst as cst
from libcst.metadata import PositionProvider
from libcst._position import CodePosition
from collections import OrderedDict

from safecoder.utils import set_logging, set_seed, get_cp_args
from safecoder.constants import PRETRAINED_MODELS, CHAT_MODELS, CWES_TRAINED, NEW_EVALS, NOT_TRAINED
from safecoder.evaler import EvalerCodePLM, EvalerCodeFT, EvalerOpenAI, EvalerChat

###
from glob import glob
from InstructorEmbedding import INSTRUCTOR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
###

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_type', type=str, choices=['trained', 'trained-new', 'not-trained'], default='trained')
    parser.add_argument('--sec_prompting', type=str, choices=['none', 'generic', 'specific'], default='none')
    parser.add_argument('--vul_type', type=str, default=None)

    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_samples_per_gen', type=int, default=20)
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=0.95)

    parser.add_argument('--experiments_dir', type=str, default='../experiments/sec_eval')
    parser.add_argument('--data_dir', type=str, default='../data_eval/sec_eval')
    parser.add_argument('--model_dir', type=str, default='../trained')

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    assert args.num_samples % args.num_samples_per_gen == 0
    if args.model_name in ('octocoder', 'llama2-13b-chat', 'codellama-13b-chat'):
        args.num_samples_per_gen = 10
    args.output_dir = os.path.join(args.experiments_dir, args.output_name, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)

    return args

def codeql_create_db(info, src_dir, db_dir):
    if info['language'] == 'py':
        cmd = '../codeql/codeql database create {} --quiet --language=python --overwrite --source-root {}'
    elif info['language'] == 'c':
        cmd = '../codeql/codeql database create {} --quiet --language=cpp --overwrite --command="make -B" --source-root {}'
    elif info['language'] in ('js', 'jsx'):
        cmd = '../codeql/codeql database create {} --quiet --language=javascript --overwrite --source-root {}'
    elif info['language'] == 'rb':
        if 'use_gemspec' in info and info['use_gemspec']:
            cmd = '../codeql/codeql database create {} --quiet --language=ruby --overwrite --command="gem build" --source-root {}'
            src_dir = os.path.dirname(src_dir)
        else:
            cmd = '../codeql/codeql database create {} --quiet --language=ruby --overwrite --source-root {}'
    elif info['language'] == 'go':
        cmd = '../codeql/codeql database create {} --quiet --language=go --overwrite --source-root {}'
    elif info['language'] == 'java':
        cmd = '../codeql/codeql database create {} --quiet --language=java --overwrite --command="bash compile_java.sh" --source-root {}'
    else:
        raise NotImplementedError()

    cmd = cmd.format(db_dir, src_dir)
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)

def codeql_analyze(info, db_dir, csv_path):
    if info['language'] in ('py', 'c', 'js', 'jsx', 'java', 'go', 'rb'):
        cmd = '../codeql/codeql database finalize {}'
        cmd = cmd.format(db_dir)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmd = '../codeql/codeql database analyze {} {} --quiet --format=csv --output={} --additional-packs={}'
        cmd = cmd.format(db_dir, info['check_ql'], csv_path, os.path.expanduser('~/.codeql/packages/codeql/'))
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

def filter_cwe78_fps(src_dir, csv_path):
    with open(csv_path) as csv_f:
        lines = csv_f.readlines()
    shutil.copy2(csv_path, csv_path+'.fp')
    with open(csv_path, 'w') as csv_f:
        for line in lines:
            row = line.strip().split(',')
            if len(row) < 5: continue
            out_src_fname = row[-5].replace('/', '').strip('"')
            out_src_path = os.path.join(src_dir, out_src_fname)
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

def eval_scenario(args, evaler, vul_type, scenario, demonstration_set, demonstration_set_embeddings, model):
    data_dir = os.path.join(args.data_dir, vul_type, scenario)
    output_dir = os.path.join(args.output_dir, vul_type, scenario)
    os.makedirs(output_dir)

    with open(os.path.join(data_dir, 'info.json')) as f:
        info = json.load(f)
        postprocess_path = os.path.join(data_dir, 'postprocess.py')
        if os.path.exists(postprocess_path):
            with open(postprocess_path) as f1:
                info['postprocess'] = f1.read()
    with open(os.path.join(data_dir, 'file_context.'+info['language'])) as f:
        file_context = f.read()
    with open(os.path.join(data_dir, 'func_context.'+info['language'])) as f:
        func_context = f.read()

    # SecCoder implementation start
    prompt = func_context
    # A custom instruction is used to create better embeddings for the demonstration
    # set and the prompt 
    task_objective_prompt = 'Represent the code for retrieving similar code snippets:'

   # task_objective_prompt = 'Represent the code snippet used as a prompt for code generation LLM for completion'
    prompt_embedding = model.encode([[task_objective_prompt, prompt]])

    # Find the best matching complete secure code example for a given prompt
    similarities = cosine_similarity(prompt_embedding, demonstration_set_embeddings)
    best_match_index = np.argmax(similarities)

    # Insert the selected code snippet at the beginning of the prompt before any imports.
    # To do this, the code is inserted before file_context using the template specified in
    # the paper.
    if info['language'] == 'py':
        formatted_snippet = f'\"\"\"\n```\n{demonstration_set[best_match_index]}\n```\n\"\"\"\n'
    elif info['language'] == 'c':
        formatted_snippet = f'#if 0\n```\n{demonstration_set[best_match_index]}\n```\n#endif\n'
    else:
        formatted_snippet = ""

    file_context = formatted_snippet + file_context

    # SecCoder implementation end

    output_srcs, non_parsed_srcs = evaler.sample(file_context, func_context, info)

    for srcs, name in [(output_srcs, 'output_srcs'), (non_parsed_srcs, 'non_parsed_srcs')]:
        src_dir = os.path.join(output_dir, name)
        os.makedirs(src_dir)
        for i, src in enumerate(srcs):
            findex = f'{str(i).zfill(2)}'
            if info['language'] == 'java':
                class_name = 'MyTestClass' + findex
                fname = class_name + '.' + info['language']
                src = src.replace('public class MyTestClass', 'public class {}'.format(class_name), 1)
            else:
                fname = findex + '.' + info['language']
            with open(os.path.join(src_dir, fname), 'w') as f:
                f.write(src)
        if name == 'output_srcs':
            if info['language'] == 'c':
                shutil.copy2('Makefile.c', os.path.join(src_dir, 'Makefile'))
            elif info['language'] == 'java':
                with open('compile_java.sh') as f:
                    makefile = f.read()
                makefile = makefile.replace('CLASS_PATH', get_cp_args(info))
                with open(os.path.join(src_dir, 'compile_java.sh'), 'w') as f:
                    f.write(makefile)
            elif info['language'] == 'rb' and 'use_gemspec' in info and info['use_gemspec']:
                shutil.copy2('test.gemspec', output_dir)

    vuls = set()
    if len(output_srcs) != 0:
        src_dir = os.path.join(output_dir, 'output_srcs')
        csv_path = os.path.join(output_dir, 'codeql.csv')
        db_dir = os.path.join(output_dir, 'codeql_db')
        codeql_create_db(info, src_dir, db_dir)
        codeql_analyze(info, db_dir, csv_path)
        if vul_type == 'cwe-078' and info['language'] == 'py':
            filter_cwe78_fps(src_dir, csv_path)
        with open(csv_path) as csv_f:
            reader = csv.reader(csv_f)
            for row in reader:
                if len(row) < 5: continue
                src_fname = row[-5].split('/')[-1]
                vuls.add(src_fname)

    d = OrderedDict()
    d['vul_type'] = vul_type
    d['scenario'] = scenario
    d['total'] = len(output_srcs)
    d['sec'] = len(output_srcs) - len(vuls)
    d['vul'] = len(vuls)
    d['non_parsed'] = len(non_parsed_srcs)
    d['model_name'] = args.model_name
    d['temp'] = args.temp

    return d

def eval_all(args, evaler, vul_types):
    
    # SecCoder implementation begin

    # Create a demonstration set by parsing the SVEN training dataset and extracting only 
    # secure implementations.
    cwe_examples_folder = "../data_train_val/train"
    demonstration_set = []

    # Get all JSON files in the folder
    file_names = ["sec-desc.jsonl", "sec-new-desc.jsonl"]
    json_files = [os.path.join(cwe_examples_folder, f) for f in file_names]
    
    # There are 9 CWE categories, and from each cateogry examples are included in the 
    # demonstration dataset. The number of included examples can be experimented with.
    for file_path in json_files:
        with open(file_path, 'r') as f:
            # Process each line (function entry) in the JSON file
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Fields like 'func_name', 'func_src_before', etc. are not needed
                    # since we are only interested in including complete secure examples
                    secure_code = data["func_src_after"]
                    demonstration_set.append(secure_code)
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Skipping invalid entry in {file_path}: {e}")
                    continue

    model = INSTRUCTOR('hkunlp/instructor-xl')
    # A custom instruction is used to create better embeddings for the demonstration
    # set and the prompt 
    task_objective_demonstration = 'Represent the code snippet for retrieval:'
    demonstration_set_instruction = [[task_objective_demonstration, snippet] for snippet in demonstration_set]
    demonstration_set_embeddings = model.encode(demonstration_set_instruction)

    # SecCoder implementation end


    for vul_type in vul_types:
        output_dir = os.path.join(args.output_dir, vul_type)
        data_dir = os.path.join(args.data_dir, vul_type)
        os.makedirs(output_dir)

        with open(os.path.join(output_dir, 'result.jsonl'), 'w') as f:
            for scenario in list(sorted(os.listdir(data_dir))):
                d = eval_scenario(args, evaler, vul_type, scenario, demonstration_set, demonstration_set_embeddings, model)
                s = json.dumps(d)
                args.logger.info(s)
                f.write(s+'\n')

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args, None)
    set_seed(args.seed)
    args.logger.info(f'args: {args}')

    if args.model_name in CHAT_MODELS:
        evaler = EvalerChat(args)
    elif args.model_name in PRETRAINED_MODELS:
        evaler = EvalerCodePLM(args)
    elif args.model_name.startswith(('gpt-3.5', 'gpt-4')):
        evaler = EvalerOpenAI(args)
    else:
        evaler = EvalerCodeFT(args)

    if args.vul_type is not None:
        vul_types = [args.vul_type]
    elif args.eval_type == 'not-trained':
        vul_types = NOT_TRAINED
    elif args.eval_type == 'trained-new':
        vul_types = NEW_EVALS
    else:
        vul_types = CWES_TRAINED

    eval_all(args, evaler, vul_types)

if __name__ == '__main__':
    main()