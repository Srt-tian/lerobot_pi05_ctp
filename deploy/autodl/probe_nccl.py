"""Validate real four-GPU collectives before FSDP training."""
import os
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl')
assert dist.get_world_size() == 4
for dtype in (torch.float32, torch.bfloat16):
    for n in (1, 1048576):
        x = torch.full((n,), rank + 1, device='cuda', dtype=dtype)
        dist.all_reduce(x)
        assert bool((x == 10).all())
        src = torch.full((n,), rank + 1, device='cuda', dtype=dtype)
        gathered = torch.empty((4 * n,), device='cuda', dtype=dtype)
        dist.all_gather_into_tensor(gathered, src)
        for r in range(4):
            assert bool((gathered[r * n:(r + 1) * n] == r + 1).all())
        src = torch.full((4 * n,), rank + 1, device='cuda', dtype=dtype)
        reduced = torch.empty((n,), device='cuda', dtype=dtype)
        dist.reduce_scatter_tensor(reduced, src)
        assert bool((reduced == 10).all())
torch.cuda.synchronize()
print('COLLECTIVES_VERIFIED', rank, torch.cuda.nccl.version(), flush=True)
dist.destroy_process_group()
