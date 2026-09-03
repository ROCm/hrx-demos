# Experiment log

All execution claims below use `iree-test-loom` or `iree-benchmark-loom` with
the default pipeline. HIP is used only for the hipBLASLt vendor artifact.
Exploratory single-run timings locate mechanisms; the final result alone uses
five-run block statistics.

| Step | Hypothesis / source form | Result | Disposition |
| --- | --- | --- | --- |
| 0 | Spike 4 all-High K64 anchor | Correct; about 60.56 us | Baseline. |
| 1 | High K32 dynamic rank-3 LDS ring | Rejected by `AMDGPU/023`; dynamic workgroup view must be rank 2 | Re-express stage choice as byte offsets into rank-2 views. |
| 2 | High K32 rank-2 LDS ring | Correct; 61.588 us; 120 VGPR, 30 SGPR, no spills | Schedule report exposes two 38-read full drains per steady loop. |
| 3 | Schedule-free register-only `low.invoke` | Correct; approximately 58.8 us; full drains remain | Boundary works but ordinary scheduling is insufficient. |
| 4 | Lock all 12 structured-loop WMMAs | Correct; approximately 83.6 us; 48 bytes spill storage and 59 scratch packets | Reject broad locking in the cyclic loop. |
| 5 | Sparse locked calls | Best exploratory subset (ordinals 2,3,4,5) approximately 57.26 us; locking 1 and 2 together spills | Useful localization, not a maintained schedule. |
| 6 | Dead locked scalar fences after loads/WMMAs | Approximately 61.8 us; fences are dead-code eliminated | Blind alley; retained tool documents why zero-value ordering tokens are not a contract. |
| 7 | `scf.for ... unroll(%thirty_two)` | Correct; approximately 54.7 us; loop full drains become 95 partial waits | Smoking-gun confirmation of loop-entry drain cost. |
| 8 | `schedule(interleaved)` on unrolled loop | Same schedule/performance as linear | No additional information. |
| 9 | `schedule(recurrence)` on unrolled loop | 1,280 bytes spill storage, 256 VGPR, approximately 64.56 us | Reject. |
| 10 | Peel final K32 tile | Removes 5 global loads, 5 LDS stores, 1 barrier, and unused fragment reads | Keep. Structured peeled loop without full unroll hits a tied-result allocator failure; record as compiler ergonomics. |
| 11 | Reorder independent WMMAs to `0,3,1,4,2,5` | High-only result 53.811 us in a fresh run | Keep as an explicit source scheduling hint. |
| 12 | One locked register-only helper after full unroll | No spills; useful LDS-read/WMMA wavefront retained; 144 VGPR | Keep. This is the narrow microkernel boundary intended by `low.invoke`. |
| 13 | Prepared-Low direct publication map copied into High | Fast (about 45 us) but nonuniform row witness fails: physical output 16 receives row 32 | Reject as semantically invalid. Uniform input had hidden the permutation. |
| 14 | High-fragment publication map `wave_m*32 + {0,16}` | All correctness/sanitizer gates pass; five-run 45.341 us versus vendor 44.601 us; upper ratio CI 1.0221 | Accept for this bounded cell. |

The final artifact is faster than the original High route without reproducing
the incumbent's loop byte-for-byte. Faster trumps schedule congruence here,
but the loop-entry wait behavior remains a compiler handoff because it forces
code expansion and will matter for less specialization-friendly kernels.
