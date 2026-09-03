# Publication experiment

Loom's generic fragment-result publication is correct, but the fresh
High-only candidate measures 53.811 us. Direct all-lane 16-bit stores cut the
cell to about 45 us even though the generated scalar-address stores carry many
conservative `vscnt(0)` waits. This confirms Spike 3's conclusion that the
publication protocol, not only the GEMM loop, is load-bearing.

The first direct map was copied from the exact prepared-Low oracle and passed
uniform checks. The new row witness immediately rejected it: physical output
index 16 held `0.5` (row 32) instead of `0.25` (row 16). The distinction is:

```text
prepared-Low oracle: row = wave_m * 16 + accumulator_group * 32 + lane_low
High fragments:      row = wave_m * 32 + accumulator_group * 16 + lane_low
```

The corrected High map passes the minimum and complete nonuniform CPU oracle.
This is a useful warning for future ports: an exact native accumulator map is
not automatically the semantic map of a High fragment carrier, even when both
lower to the same WMMA instruction.
