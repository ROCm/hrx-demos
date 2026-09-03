# Loop-carried wait experiment

The structured K32 ring emits the right arithmetic and memory counts but the
default wait plan batches 38 LDS reads and drains them at the loop boundary.
The implementation in `planning/wait_plan.c` recognizes an SSA dependency
whose producer and consumer cross a natural-loop entry, relocates it to a
preheader slot, and later calls the wait planner with `target_count=0`.

Full unrolling is the decisive counterfactual. The dataflow and High memory
operations are unchanged, but the cyclic boundary disappears. The report then
contains 95 partial waits and performance improves by roughly 7 us. This
isolates the lost overlap to the conservative loop-entry contract rather than
tile shape, instruction availability, or generic “compiler quality.”

The exact-shape JIT is allowed to choose full unrolling, so the workaround is
valid for this cell. The compiler handoff still matters because wholesale
unrolling produces 39 KiB of code and 40.7 ms median bytecode-to-HSACO latency.
