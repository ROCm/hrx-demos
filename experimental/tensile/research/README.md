# Research index

Research is organized as numbered spikes. Each spike freezes a question and
retains enough state to reproduce or invalidate its findings.

- [`spike-001-primitive-gemm/`](spike-001-primitive-gemm/) — establish the
  incumbent-selection, recipe-extraction, Loom-compilation, correctness, and
  measurement loop for primitive FP16 GEMM.
- [`spike-002-schedule-congruence/`](spike-002-schedule-congruence/) — define
  the Radeon shape/datatype basket, reconstruct exact incumbent routes, and
  prove accelerated instruction representability.
- [`spike-003-native-upward/`](spike-003-native-upward/) — recover schedules
  from shipped native code upward, including the gfx12 parity Low motif, the
  gfx11 allocation/scheduling reproducer, blind alleys, access-sanitizer
  usability reports, and proposed compile-report knowledge.
- [`spike-004-default-pipeline-microkernels/`](spike-004-default-pipeline-microkernels/)
  — test raising retained Low schedule oracles into High Loom kernels through
  `low.invoke`; retain the accepted gfx12 performance witness, upstream #513
  validation of family-generic locked fragments, and exact packets for the
  unsupported or still-unverified boundaries.
- [`spike-005-gfx11-structured-recovery/`](spike-005-gfx11-structured-recovery/)
  — recover the gfx11 FP16 cell through structured High memory/control flow
  plus one register-only locked WMMA helper; retain the loop-wait smoking gun,
  publication-map correction, full nonuniform and access-sanitizer evidence,
  and a default-pipeline 1.0166x result versus hipBLASLt solution 1675.
- [`tools/`](tools/) — small explicit probes used by the spikes.

Generated binaries and bulky raw artifacts live in ignored `artifacts/`
directories. Small JSON, text reports, source, disassembly, and manifests are
committed when they are evidence.

The native-symbol inspector handles Tensile code objects whose ELF function
symbols have size zero by bounding a requested kernel at the next external
text symbol. Its JSON report keeps the artifact hash, bounds, target, mnemonic
histogram, and coarse memory/synchronization counts without committing the
multi-megabyte unbundled HSACO or disassembly.

For Loom command-line work, consult each tool's `--agents_md` output before
using ordinary `--help`; it contains the agent-oriented compile, report, and
evidence workflow. The spike uses `loom-compile` reports as the first schedule
oracle and disassembly as the native check.

The governing provider design is [`../docs/design/loom-blas.md`](../docs/design/loom-blas.md).
