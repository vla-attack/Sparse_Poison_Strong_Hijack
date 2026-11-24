"""Single physical GPU: first-cup calibration, export, 80 rollouts and comparisons."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from experiments.vla_safety.dual_lora_eval_protocol import (
    candidate_info,checkpoint_info,expected_scenes,prepare_evaluation,read_json,sha,validate_results,
)


def environment(repo):
    root=repo.parent
    env=dict(os.environ)
    env['PYTHONPATH']=':'.join(map(str,(repo,root/'torch_environment_bundle_20261011/src/openvla-oft',
        root/'envs/torch/local-src/LIBERO-master',root/'envs/torch/local-src/transformers-openvla-oft-main/src')))+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['LD_LIBRARY_PATH']=str(root/'runtime_libs/glvnd_egl/usr/lib/x86_64-linux-gnu')+':/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env.update(LIBERO_CONFIG_PATH=str(root/'libero_eval_config'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TF_CPP_MIN_LOG_LEVEL='3',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2')
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    return env


def preflight(repo,output,model):
    parent=repo/'experiments/vla_safety/artifacts/clean150k_pcgrad_cup_dual10k_train_20261102T051038_743'
    second=repo/'experiments/vla_safety/artifacts/grouped_fusion_pcgrad_two_tasks_20261103'
    source=Path((parent/'checkpoint.txt').read_text().strip())
    data=parent/'optimized_rlds'
    first_baseline=parent/'evaluation_before_reoptimization'
    reference=second/'evaluation_quick_task0_task1'
    source_identity=checkpoint_info(source,10000)
    first_manifest=read_json(first_baseline/'evaluation_manifest.json')
    reference_manifest=read_json(reference/'evaluation_manifest.json')
    if source_identity!=first_manifest['checkpoint']:raise ValueError('Wrong source 10k checkpoint')
    cup=candidate_info(data/'candidate/candidate.json')
    if cup['candidate_sha256']!=first_manifest['candidate_sha256']:raise ValueError('Wrong first-stage cup')
    if cup['candidate_sha256']==reference_manifest['candidate_sha256']:raise ValueError('First and second cups must differ')
    if any(read_json(folder/'evaluation_validation.json')['status']!='passed' for folder in (first_baseline,reference)):
        raise ValueError('Existing comparisons must be complete and audited')
    if reference_manifest['protocol']['task_ids']!=[0,1] or reference_manifest['protocol']['state_ids']!=list(range(1,21)):
        raise ValueError('Expected task0/task1 state1-20 comparison')
    calibration=read_json(second/'calibration/coefficients.json')
    if calibration['source_model']!=source_identity or calibration['method']!='pcgrad' or calibration['world_size']!=1:
        raise ValueError('Expected the same frozen model and single-GPU PCGrad calibration')
    for name,digest in reference_manifest['rollout_source_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Physical rollout code changed: '+name)
    if output.resolve()==second.resolve() or model.resolve() in (source.resolve(),Path(reference_manifest['checkpoint']['checkpoint']).resolve()):
        raise ValueError('Use independent output and model directories')
    return dict(parent=parent,second=second,source=source,data=data,first_baseline=first_baseline,
        reference=reference,calibration=calibration,candidate=data/'candidate/candidate.json',source_identity=source_identity)


def align_calibration_bank(raw_path,template_path,output,expected_template_sha):
    """Keep sampling/labels/reference normalizers fixed; change only cup pixels/ROIs."""
    import torch
    torch.set_num_threads(2)
    if sha(template_path)!=expected_template_sha:raise ValueError('Completed second-cup calibration bank changed')
    required=dict(raw_bank_sha256=sha(raw_path),reference_template_bank_sha256=sha(template_path))
    metadata=output.with_suffix('.json')
    if output.exists():
        saved=read_json(metadata)
        if any(saved['identity'].get(key)!=value for key,value in required.items()) or saved['bank_sha256']!=sha(output):
            raise ValueError('Aligned first-cup bank identity changed')
        return
    raw=torch.load(raw_path,map_location='cpu',weights_only=True)
    template=torch.load(template_path,map_location='cpu',weights_only=True)
    same_keys=('input_ids','attention_mask','labels','actions','proprio')
    proof=dict(clean_tensor_inputs_identical=True,trigger_labels_proprio_tokens_identical=True,
               trigger_references_identical=True,trigger_pixel_frames_different=0,trigger_roi_frames_different=0)
    records=[]
    def key(sample):
        row=sample['record'];return row['task_id'],row['trajectory_step'],row['frame_id']
    for split in ('fit','selection'):
        for source in ('clean','trigger'):
            indexed={key(sample):sample for sample in raw['bank'][split][source]}
            reference=template['bank'][split][source]
            if set(indexed)!={key(sample) for sample in reference}:raise ValueError('First/second calibration frame selection differs')
            reordered=[]
            for sample in reference:
                first=indexed[key(sample)]
                for name in same_keys:
                    if not torch.equal(first['batch'][name],sample['batch'][name]):
                        raise ValueError('Calibration labels/proprio/tokens differ: '+source+'/'+name)
                if source=='clean':
                    if not torch.equal(first['batch']['pixel_values'],sample['batch']['pixel_values']):
                        raise ValueError('Clean pixels differ across calibration banks')
                else:
                    proof['trigger_pixel_frames_different']+=int(not torch.equal(first['batch']['pixel_values'],sample['batch']['pixel_values']))
                    proof['trigger_roi_frames_different']+=int(any(not torch.equal(first['batch'][name],sample['batch'][name]) for name in ('trigger_cup','trigger_bowl')))
                    for name in ('trigger_reference','trigger_reference_valid'):
                        first['batch'][name]=sample['batch'][name].clone()
                reordered.append(first);records.append(first['record'])
            raw['bank'][split][source]=reordered
    if not proof['trigger_pixel_frames_different']:raise ValueError('First-cup images are unexpectedly identical to second-cup images')
    raw['identity'].update(required)
    raw['identity']['reference_policy']='Identical frozen-model reference/reference_valid tensors and sample order to completed second-cup calibration; own first-cup pixels and cup/bowl ROIs'
    temporary=output.with_suffix('.tmp');torch.save(raw,temporary);temporary.replace(output)
    metadata.write_text(json.dumps(dict(identity=raw['identity'],records=records,control_checks=proof,bank_sha256=sha(output)),indent=2)+'\n')
    print('CALIBRATION_CONTROLS',json.dumps(proof),flush=True)


def run(args):
    repo=Path(__file__).resolve().parents[2]
    config=preflight(repo,args.output,args.model_dir)
    if args.check:
        print(json.dumps(dict(check='passed',first_candidate_sha256=candidate_info(config['candidate'])['candidate_sha256'],
            source=str(config['source']),calibration_steps=config['calibration']['steps'],tasks=[0,1],states=list(range(1,21)),
            physical_gpus=1,logical_shards=4,execution='sequential'),indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    env=environment(repo)
    if ',' in env.get('CUDA_VISIBLE_DEVICES','0'):raise ValueError('Expose exactly one physical GPU to this experiment')
    python=sys.executable
    stage='preflight'
    def mark(value):
        nonlocal stage
        stage=value;(args.output/'stage.txt').write_text(value+'\n')
        print('STAGE',datetime.now(timezone.utc).isoformat(),value,flush=True)
    def execute(command,log=None):
        print('EXEC',json.dumps(list(map(str,command))),flush=True)
        if log is None:
            subprocess.run(list(map(str,command)),cwd=repo,env=env,check=True)
        else:
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:
                result=subprocess.run(list(map(str,command)),cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text(errors='replace')[-8000:],flush=True)
                raise subprocess.CalledProcessError(result.returncode,command)
    try:
        mark('gpu_check')
        execute([python,'-c','import torch; assert torch.cuda.device_count()==1; print(torch.cuda.get_device_name(0))'])
        settings=config['calibration']
        experiment=dict(source_model=config['source_identity'],first_candidate=candidate_info(config['candidate']),
            existing_second_evaluation=str(config['reference']),physical_gpu_count=1,logical_episode_shards=4,
            shard_execution='sequential',cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES','0'),
            tasks=[0,1],states=list(range(1,21)),episodes_per_condition=40,calibration_steps=settings['steps'],
            calibration_learning_rate=settings['learning_rate'],calibration_seed=settings['seed'],
            output_model=str(args.model_dir.resolve()))
        (args.output/'experiment_config.json').write_text(json.dumps(experiment,indent=2)+'\n')
        snapshot=args.output/'source_snapshot';snapshot.mkdir(exist_ok=True)
        for name in ['experiments/vla_safety/run_grouped_fusion_firstcup_single.py',
            'experiments/vla_safety/compare_grouped_cup_stages.py','experiments/vla_safety/prepare_fusion_calibration_data.py',
            'experiments/vla_safety/calibrate_grouped_lora_fusion.py','experiments/vla_safety/training/grouped_lora_fusion.py',
            'experiments/vla_safety/training/dual_lora_export.py','experiments/vla_safety/dual_lora_eval_protocol.py',
            'experiments/vla_safety/eval_fusion_subset.py']:
            shutil.copy2(repo/name,snapshot/Path(name).name)
        mark('first_cup_calibration_cache')
        execute([python,'-u','-m','experiments.vla_safety.prepare_fusion_calibration_data',
            '--checkpoint',config['source'],'--trigger-root',config['data'],'--clean-root',repo.parent/'modified_libero_rlds',
            '--baseline',config['first_baseline'],'--output',args.output/'calibration_bank_raw.pt'])
        mark('align_calibration_frames_and_reference_normalizers')
        align_calibration_bank(args.output/'calibration_bank_raw.pt',config['second']/'calibration_bank.pt',
                               args.output/'calibration_bank.pt',settings['bank_sha256'])
        coefficients=args.output/'calibration/coefficients.json'
        mark('first_cup_coefficient_calibration_and_export')
        if not coefficients.exists():
            execute([python,'-u','-m','experiments.vla_safety.calibrate_grouped_lora_fusion',
                '--checkpoint',config['source'],'--bank',args.output/'calibration_bank.pt','--output',args.output/'calibration',
                '--steps',settings['steps'],'--method','pcgrad','--learning-rate',settings['learning_rate'],
                '--clean-tolerance',settings['clean_tolerance'],'--seed',settings['seed'],'--selection-frequency',10,
                '--export-dir',args.model_dir])
        if not (args.model_dir/'dual_lora_merge_report.json').exists() or (args.model_dir/'INCOMPLETE').exists():
            raise RuntimeError('Model export incomplete; inspect calibration log before resuming')
        record=read_json(coefficients)
        info=checkpoint_info(args.model_dir,10000,config['source_identity']['base_model_path'])
        if info['trigger_group_scales']!=record['trigger_group_scales'] or info['calibration']['sha256']!=sha(coefficients):
            raise ValueError('Export and new first-cup calibration disagree')
        if record['bank_identity']['candidate']['candidate_sha256']!=candidate_info(config['candidate'])['candidate_sha256']:
            raise ValueError('Calibration did not use the first-stage cup')
        mark('verify_export_and_processor')
        execute([python,'-m','experiments.vla_safety.training.verify_grouped_fusion_export','--source',config['source'],
            '--export',args.model_dir,'--output',args.output/'export_validation.json'])
        execute([python,'-c','from transformers import AutoProcessor; import sys; '
            'p=AutoProcessor.from_pretrained(sys.argv[1],trust_remote_code=True,local_files_only=True); print(type(p).__name__)',args.model_dir])
        evaluation=args.output/'evaluation_quick_task0_task1'
        runner=repo/'experiments/vla_safety/eval_fusion_subset.py'
        mark('prepare_physical_evaluation')
        prepare_evaluation(args.model_dir,evaluation,config['candidate'],shards=4,step=10000,
            base=config['source_identity']['base_model_path'],repo=repo,task_ids=[0,1],state_ids=list(range(1,21)),runner=runner)
        env['OFT_TRIGGER_CANDIDATE']=str(evaluation/'candidate_snapshot/candidate.json')
        shared=[python,'-u',runner,'--checkpoint',args.model_dir,'--task-ids','0,1',
                '--state-ids',','.join(map(str,range(1,21))),'--shards','4']
        for shard in range(4):
            mark('single_gpu_evaluation_shard_'+str(shard))
            execute([*shared,'--output',evaluation/f'shards/gpu{shard}','--shard',shard],
                    evaluation/f'logs/gpu{shard}.log')
        mark('aggregate_and_audit')
        execute([*shared,'--output',evaluation,'--aggregate'],evaluation/'logs/aggregate.log')
        validate_results(evaluation)
        mark('same_cup_and_cross_cup_comparisons')
        execute([python,'-m','experiments.vla_safety.compare_grouped_fusion',
            '--baseline',config['first_baseline'],'--evaluation',evaluation,
            '--mapping',config['data']/'frame_mapping.json','--output',evaluation/'same_cup_baseline_comparison'])
        report=evaluation/'same_cup_baseline_comparison/RESULTS.md'
        report.write_text(report.read_text().replace('和第二轮杯子','和第一轮杯子'))
        execute([python,'-m','experiments.vla_safety.compare_grouped_cup_stages',
            '--first',evaluation,'--second',config['reference'],'--output',evaluation/'cross_cup_comparison'])
        mark('completed')
        (args.output/'exit_code.txt').write_text('0\n')
        print('FINAL_REPORT',evaluation/'cross_cup_comparison/RESULTS.md',flush=True)
    except BaseException:
        (args.output/'stage.txt').write_text('failed:'+stage+'\n')
        (args.output/'exit_code.txt').write_text('1\n')
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    run(parser.parse_args())
