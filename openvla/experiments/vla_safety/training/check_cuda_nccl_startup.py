"""Small per-GPU CUDA kernel and NCCL check before loading model weights.
Diagnoses current allocation only. Passing does not certify future GPU health.
"""
import os,json,socket,argparse,traceback
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

@record
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--expected-world-size',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--isolate-local-device',action='store_true')
 parser.add_argument('--memory-test-mib',type=int,default=0)
 parser.add_argument('--memory-test-blocks',type=int,default=8);args=parser.parse_args()
 if args.memory_test_mib<0 or args.memory_test_blocks<1:parser.error('Invalid memory test size')
 original_local=int(os.environ['LOCAL_RANK'])
 if args.isolate_local_device:
  from experiments.vla_safety.training.cuda_device_mapping import isolate_local_cuda_device
  isolate_local_cuda_device()
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);args.output.mkdir(parents=True,exist_ok=True)
 result=dict(rank=rank,local_rank=local,original_local_rank=original_local,world_size=world,hostname=socket.gethostname(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),stage='device_initialization',passed=False)
 try:
  assert world==args.expected_world_size,(world,args.expected_world_size)
  torch.set_num_threads(1);torch.cuda.set_device(local)
  result['gpu_name']=torch.cuda.get_device_name(local)
  props=torch.cuda.get_device_properties(local);result['gpu_uuid']=str(getattr(props,'uuid','see gpus.csv'))
  print('GPU_CHECK_BEGIN '+json.dumps(result),flush=True)
  if args.memory_test_mib:
   result['stage']='cuda_memory_write_read'
   blocks=[torch.empty(args.memory_test_mib*1024*1024,dtype=torch.uint8,device=f'cuda:{local}')
           for _ in range(args.memory_test_blocks)]
   for pattern in (0x55,0xAA):
    for block in blocks:block.fill_(pattern)
    torch.cuda.synchronize(local)
    for block in blocks:
     assert bool((block==pattern).all()),'GPU memory pattern mismatch'
    torch.cuda.synchronize(local)
   result['memory_test_mib']=args.memory_test_mib*args.memory_test_blocks
   del blocks,block
   torch.cuda.empty_cache()
  result['stage']='cuda_allocation_and_kernel'
  x=torch.ones((256,256),device=f'cuda:{local}')
  for _ in range(3):x=(x@x)/256
  torch.cuda.synchronize(local);assert bool(torch.isfinite(x).all()) and bool((x==1).all())
  result['stage']='nccl_allreduce_broadcast_barrier'
  dist.init_process_group('nccl',timeout=timedelta(seconds=90))
  value=torch.tensor([float(rank+1)],device=f'cuda:{local}');dist.all_reduce(value)
  assert float(value)==world*(world+1)/2
  value.fill_(7. if rank==0 else 0.);dist.broadcast(value,src=0);assert float(value)==7.
  dist.barrier(device_ids=[local]);torch.cuda.synchronize(local)
  result.update(stage='completed',passed=True)
  (args.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
  print('GPU_CHECK_PASS '+json.dumps(result),flush=True)
  dist.destroy_process_group()
 except BaseException as exc:
  result.update(error=repr(exc),traceback=traceback.format_exc())
  (args.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2))
  print('GPU_CHECK_FAILED '+json.dumps(result),flush=True)
  raise
if __name__=='__main__':main()
