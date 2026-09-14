# AutoDL vGPU NCCL query compatibility

The CUDA runtime's cudaGetDriverEntryPoint returns success and a pointer but
leaves the query status untouched on this host. NCCL2.28.9 treats this as a
missing symbol and segfaults at its first barrier. A direct driver
cuGetProcAddress_v2 probe returns both a valid pointer and success; a nonexistent
symbol returns no pointer and SYMBOL_NOT_FOUND.

Use a task-scoped NCCL2.28.9 build with nccl_vgpu_query.patch. This resolves the
exact API version declared by NCCL through driver v2 and preserves errors and
null checks. The system driver and shared Python environment stay untouched.

Source: https://codeload.github.com/NVIDIA/nccl/tar.gz/refs/tags/v2.28.9-1
Archive SHA256: f349860336c6b7fb97b22bed9c729142f3531a0e82826c1204d01e44af8b9cb9
Build with CUDA12.4, make -j8 src.build
NVCC_GENCODE='-gencode=arch=compute_89,code=sm_89'.
Apply patch with patch -p1 in extracted source before building.
Load the resulting libnccl.so.2 through LD_PRELOAD only for this task.
Verify all-reduce, all-gather and reduce-scatter on all4 GPUs before training.
Sanitized runtime logs and the resulting library hash belong in the run ledger.

Validated library SHA256: 93b292d67360a457764149d3aa192a7f4886b3d0e9b64034894055d24ca4ebdf
Preload system libstdc++.so.6 before libnccl; disable CUMEM and CUMEM_HOST.
Four ranks passed FP32/BF16 collectives at1 and1048576 elements.
