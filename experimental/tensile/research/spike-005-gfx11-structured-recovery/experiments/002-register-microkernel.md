# Register-only microkernel experiment

A schedule-free `low.invoke` demonstrated that High fragment values can cross
the helper ABI and specialize for `gfx11-generic`. It did not retain enough
source order to solve the LDS/WMMA issue. Locking every helper call inside the
cyclic structured loop overconstrained allocation and introduced spill
traffic.

After exact-shape unrolling, one locked helper definition is sufficient. Each
call remains at its authored position, values remain in registers, and the
final artifact has no private memory. The assembly wavefront begins with a
partial LDS wait, a group of A reads, another partial wait, then interleaves B
reads and the `0,3,1,4,2,5` WMMA sequence. Low therefore acts as a microkernel
surface, not an assembler for the whole GEMM.

`insert_gfx11_schedule_fences.py` records a blind alley: zero-input helpers
whose result is unused are dead-code eliminated and cannot serve as ordering
tokens. A future High scheduling construct needs explicit semantics, not an
accidental live-value trick.
