# Experiment index

This index is the shortest path through the retained schedule-recovery state.
Generated HSACOs, disassembly, compiler reports, and traces are under
`artifacts/`; compact timing and schedule summaries are under `results/` and
`schedules/`. The transformation scripts are intentionally narrow and fail if
their expected prepared-Low pattern changes.

## Positive witnesses

| Question | Source or tool | Evidence | Result |
| --- | --- | --- | --- |
| Can gfx12 express TensileLite's B64+permute A-fragment ABI? | `research/tools/rewrite_gfx12_*_b64_permute.py` and the final gfx12 Low source | `artifacts/gfx12-uniform-bpad-scalar-saddr-native-rhs-lhs-first/` | Yes; eight B64 reads and sixteen permutes per K=16 half. |
| Can a manual gfx12 loop reach runtime parity? | `low/gfx12-gemm-double-buffer-pgr2-ring-native-split-exact-fenced-svw4-uniform-bpad-scalar-saddr-ring-native-rhs-lhs-first.loom` | paired and ATT files prefixed `gfx1201-uniform-bpad-...-lhs-first` and `gfx1201-att-current` | Yes for the 1024-cubed FP16 anchor: 1.0438x, upper CI 1.0474. |
| Can gfx11 contain the native primitive instruction budget without spills? | `tools/hoist_gfx11_native_lds_addresses.py` | `artifacts/gfx11-native-lds-address-hoisted/`, paired prefix `gfx1100-native-lds-address-hoisted` | Yes: about 133 dynamic descriptors/K32 versus native 134, zero private memory. |
| Can Loom source express gfx11 cross-iteration PLR? | `tools/make_gfx11_cross_iteration_plr.py`, High and prepared-Low cross-iteration sources | `artifacts/gfx11-cross-iteration-plr/` | Yes semantically; allocation introduces four scratch values. |
| Can a late compiler pass remove the gfx11 fragment backedge braid? | HRX `loom-blas/gfx11-structural-loop-recolor` at `0b70e4379` | `artifacts/gfx11-cross-iteration-plr-wide-relocation-trace/`, `results/gfx1100-loop-structural-recolor-summary.json` | Yes for the copies and repeated correctness, but not timing: all 40 recurrent fragment moves disappear and the paired result remains 1.6285x. |
| Can a prepared-Low gfx11 motif exactly reproduce the native steady-state schedule? | Terminal `low/gfx11-native-k32-...-swapped-native-epilogue.loom` plus the gated HRX branch at `b422b5056` | `schedules/gfx1100-loom-terminal-exact-branch-steady-state.json`, `results/gfx1100-terminal-exact-branch-schedule-comparison.json` | Yes: 134 instructions and identical category counts; only equivalent branch polarity differs. |
| Does the exact gfx11 schedule recover runtime parity? | Same terminal source | paired/full-correctness files prefixed `gfx1100-native-k32-wgm8-swapped-native-epilogue-exact-branch` | Yes: 37.721 us versus 44.320 us, ratio 0.8511 with upper CI 0.8553; all 1024-cubed outputs pass. |
| Is the gfx11 scalar epilogue address map legal under native access instrumentation? | `tools/use_gfx11_native_high_epilogue.py --keep-mma-roles` | `artifacts/gfx11-high-native-epilogue-address-witness/` | Yes for the exact address formulas. The complete paired High form is separately rejected by fragment-role validation. |
| Are accelerated matrix types representable? | spike-002 `gfx12-accelerated-type-witness.loom` and `gfx11-i4-type-witness.loom` | `spike-002-schedule-congruence/results/accelerated-types.json` | Native BF16/I8/I4 and gfx12 FP8/BF8 instruction witnesses compile. |

## Gfx12 progression and blind alleys

The principal stages can be reconstructed from these source families:

1. `loom/gemm-f16-f32-mt128x128x32-gfx12-scalar-acc.loom` — High baseline.
2. `research/tools/rewrite_gfx12_lhs_b64_permute.py` and later compact,
   prefetch, double-buffer variants — manual fragment packing.
3. `research/tools/make_gfx12_pgr2_ring_low.py` — explicit load ring.
4. `research/tools/interleave_gfx12_pgr2_native_split.py` — native half-tile
   issue grouping.
5. spike-003 `pad_gfx12_b_lds*.py`, `use_gfx12_scalar_saddr_ring.py`,
   `use_gfx12_native_rhs_ownership.py`, and `fence_gfx12_lhs_before_rhs.py` —
   bank layout, address form, ownership, and final order.

Failed or diagnostic-only paths are preserved beside the successful path:

- scalar/scatter A transposition moved cost to stores and measured roughly
  85 us;
- compact and peeled loops proved that removing High unrolling alone does not
  recover allocation or packing;
- direct-tie and unsafe partial-wait experiments either violated tied-result
  lifetime rules or produced wrong results;
- buffer-global-load variants did not beat scalar-saddr global loads;
- authored delay, reverse pack, accumulator pinning, and exact fragment pinning
  changed local placement but did not improve the final paired gate;
- skipping inferred source-reuse waits was retained only when correctness
  checks demonstrated the risk; it is not a sanctioned optimization.

The compiler-control branch `loom-blas/gfx12-wait-state-parity` (`ec40ab4f0`)
contains the schedule knobs used to separate source order from wait-state
effects. The final performance result does not require making those controls
the default.

## Gfx11 allocation triage matrix

Baseline for this table is the cross-iteration prepared Low with four spill
objects, four reloads, 16 bytes of private storage, peak pressure 151, final
VGPR allocation 160, and 71 units of branch-edge moves.

| Experiment | Retained evidence | Outcome |
| --- | --- | --- |
| Reuse invariant LDS store addresses | `tools/hoist_gfx11_cross_iteration_store_addresses.py`, artifact suffix `store-address-hoisted` | Regressed to ten spills. Longer address lifetimes conflict with fragment banks. |
| Rematerialize every store index | `tools/rematerialize_gfx11_lds_store_indices.py`, artifact suffix `store-index-remat` | Five spills and substantial reload traffic. |
| Rematerialize only at the second store group | `tools/rematerialize_gfx11_lds_store_indices_local.py`, artifact suffix `store-index-local-remat` | Same five-spill regression. |
| Allocate wide intervals first | artifact suffix `wide-first` | No change: four spills. |
| Pack only x8-aligned intervals first | artifact suffix `pack-wide` | 27 spills, 108 bytes private. |
| Treat all virtual values as alignment one | artifact suffix `packed-registers` | 20 spills, 80 bytes private. |
| Pin accumulators/fragments/native banks | artifact prefixes `fixed-*` | Fixed constraints created conflicts or additional blockers. |
| Permit ISA-legal unaligned fixed banks and shift by 1/4/8 | artifacts `fixed-shifted1`, `4`, `8` | 12 spills; shift 8 reduced bytes but did not solve placement. |
| Relocate wide loop-edge intervals | `loom-blas/vector-loop-edge-relocation` (`c18abc7b9`) and artifact `wide-loop-relocation` | No change. |
| Combine relocation with all gfx11 controls | `loom-blas/gfx11-vector-loop-edge-relocation-combined` (`60b26addb`) | No change. |
| Publish the five VMEM payloads after the first three WMMAs, matching native issue position | `tools/interleave_gfx11_payload_stores.py`, source/artifact suffix `native-payload-order` | Still four spills; x8 high-water moves to `odd_first_rhs2` and branch-edge moves increase from 71 to 95. Compiler gate failed, so no timing run. |
| Apply `scf.for` unroll factor 2 | `tools/make_gfx11_cross_iteration_plr.py --unroll-factor two`, artifact suffix `unroll-two` | Four spill objects remain; reload traffic rises to 24 bytes and branch moves increase. |
| Apply `scf.for` unroll factors 4/8/full | generated source suffixes `unroll-four`, `unroll-eight`, `unrolled` | Regresses to 49/73/154 spill objects. Unrolling is not the missing schedule representation. |
| Recolor only carried aggregate results | intermediate HRX structural-recolor experiment | False win: edge copies move to `low.concat`; static instruction count is unchanged. |
| Recolor aggregate placement closures and lease-backed blockers jointly | HRX `loom-blas/gfx11-structural-loop-recolor` (`0b70e4379`) | Backedge fragment copies drop from 40 to zero; dynamic register moves drop 687 to 303. The corrected artifact passes repeated correctness, but remains 1.6285x solution 1675; prepared-Low access instrumentation is unavailable. |
| Rematerialize seven address inputs at the payload-write peak | `low/gfx11-cross-iteration-plr-split-zero-remat-address-inputs.loom`, artifact suffix `remat-address-inputs` | Scheduled pressure falls 151 to 132, but iterative repair expands to 19 spill objects/76 bytes and loses the structural recolor. |
| Substitute native raw-buffer address mode without joint SRD placement | `tools/use_gfx11_buffer_loads.py`, Low suffix `buffer-loads-dce`, artifact suffix `buffer-loads-dce` | Emits all 15 requested `buffer_load_b128` instructions and is correct, but expands to 20 spill objects/80 bytes, 124 waits, and about 131 us. |
| Lock the substituted raw-buffer source order | Low suffix `buffer-loads-dce-locked`, matching artifact suffix | No change to spills, waits, or placement. The cliff is allocation/lease composition, not scheduler freedom. |

The allocation controls and verifier bypass used for the matrix are isolated
on `loom-blas/gfx11-loop-bank-triage` (`5c4d6ef41`). They are research
instruments, not defaults. The result rejects several plausible but overly
simple remedies: register pressure reduction, alignment relaxation, one
allocation-order heuristic, and native register pinning.

The structural recoloring branch is intentionally later and separate. Its
environment gate defaults off, its compiler build succeeds, and the complete
allocation test package passes 20/20. It should be treated as a specification
for a smaller allocator change, not as a patch proposed verbatim for landing.

## Access sanitizer cases

| Case | Compile | Launch | Interpretation |
| --- | --- | --- | --- |
| gfx12 High, one 128x128x32 tile | Pass | Pass | Complete High address-map witness. |
| gfx12 High, fully unrolled 1024 cube | `BACKEND/021` after instrumentation | Not reached | Instrumentation/allocation explosion. |
| gfx11 High exact-LDS, 64x96x64, raw HIP loader | Pass | GPU page fault | Invalid sanitizer launch workflow: shadow/report runtime was not initialized. |
| gfx11 High exact-LDS, 64x96x64, AMDGPU `iree-test-loom` | Pass | Pass | Supported sanitizer workflow; numerical oracle passes with no event. |
| gfx11 High terminal scalar-address witness | Pass | Pass | Exact terminal address formulas pass access; retains original High MMA roles. |
| Prepared Low with `--pipeline=none` | Pass | Runs uninstrumented | Sanitizer is silently absent from this pipeline. |
| Prepared Low through default access pipeline | Internal lowering failure | Not reached | Missing supported prepared-Low sanitizer path. |

See `compiler-usability-access-sanitizer.md` before interpreting any Low
correctness result as a memory-legality result.

## Reproduction conventions

- `ROCR_VISIBLE_DEVICES=1` selects gfx1100; the unrelated `amd-smi` ordinal is
  0 (W7900).
- `ROCR_VISIBLE_DEVICES=2` selects gfx1201; the `amd-smi` ordinal is 1.
- `ROCR_VISIBLE_DEVICES=0` selects gfx906; the `amd-smi` ordinal is 2.
- Runtime commands use `LD_LIBRARY_PATH=<selected-rocm>/lib`.
- Prepared Low is compiled with `loom-compile --pipeline=none`; High source
  uses the normal target pipeline.
- `build/blas-probe` is the correctness-gated, public-API comparison harness.
- `research/tools/analyze_paired.py` performs deterministic bootstrap analysis.
- Native intervals are normalized with
  `research/tools/extract_amdgpu_interval.py`.
- Compile reports, not source intuition, are authoritative for spill, wait,
  pressure, and target-resource claims.
