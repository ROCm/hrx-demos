// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "loomc/loomc.h"
#include "loomc/target/amdgpu.h"

typedef struct benchmark_state_t {
  loomc_target_environment_t* target_environment;
  loomc_context_t* context;
  loomc_workspace_t* workspace;
  loomc_source_t* source;
  loomc_target_profile_t* target_profile;
  loomc_compiler_t* compiler;
  loomc_pass_program_t* pass_program;
} benchmark_state_t;

typedef struct sample_t {
  double deserialize_us;
  double compile_us;
  double emit_us;
  double total_us;
  uint64_t artifact_bytes;
} sample_t;

static uint64_t now_ns(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    perror("clock_gettime");
    exit(1);
  }
  return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}

static void print_status(loomc_status_t status) {
  char buffer[2048] = {0};
  loomc_host_size_t length = 0;
  loomc_status_format(status, sizeof(buffer), buffer, &length);
  fprintf(stderr, "%.*s\n", (int)length, buffer);
}

static void print_result_diagnostics(const loomc_result_t* result) {
  if (result == NULL) return;
  for (loomc_host_size_t i = 0; i < loomc_result_diagnostic_count(result);
       ++i) {
    const loomc_diagnostic_t* diagnostic =
        loomc_result_diagnostic_at(result, i);
    if (diagnostic == NULL) continue;
    fprintf(stderr, "%.*s: %.*s\n", (int)diagnostic->code.size,
            diagnostic->code.data, (int)diagnostic->message.size,
            diagnostic->message.data);
  }
}

static int require_successful_result(const loomc_result_t* result,
                                     const char* phase) {
  if (result != NULL && loomc_result_succeeded(result)) return 1;
  fprintf(stderr, "%s produced a failed result\n", phase);
  print_result_diagnostics(result);
  return 0;
}

static const loomc_artifact_t* find_hsaco(const loomc_result_t* result) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != NULL &&
        artifact->kind == LOOMC_ARTIFACT_KIND_EXECUTABLE &&
        loomc_string_view_equal(
            artifact->format,
            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO))) {
      return artifact;
    }
  }
  return NULL;
}

static void benchmark_state_deinitialize(benchmark_state_t* state) {
  loomc_pass_program_release(state->pass_program);
  loomc_compiler_release(state->compiler);
  loomc_target_profile_release(state->target_profile);
  loomc_source_release(state->source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
  loomc_target_environment_release(state->target_environment);
  memset(state, 0, sizeof(*state));
}

static loomc_status_t benchmark_state_initialize(benchmark_state_t* state,
                                                 const char* source_path,
                                                 const char* target) {
  memset(state, 0, sizeof(*state));
  loomc_status_t status = loomc_target_environment_create_amdgpu(
      loomc_allocator_system(), &state->target_environment);

  loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .target_environment = state->target_environment,
  };
  loomc_context_options_t context_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(context_options),
      .next = &target_options,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_context_create(&context_options, loomc_allocator_system(),
                                  &state->context);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_workspace_create(NULL, loomc_allocator_system(),
                                    &state->workspace);
  }

  loomc_source_load_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_LOAD_OPTIONS,
      .structure_size = sizeof(source_options),
      .format = LOOMC_SOURCE_FORMAT_BYTECODE,
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create_from_path(
        loomc_make_cstring_view(source_path), &source_options,
        loomc_allocator_system(), &state->source);
  }

  loomc_amdgpu_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_make_cstring_view("loom-blas-jit-benchmark"),
      .identity = {.target = loomc_make_cstring_view(target)},
  };
  if (loomc_status_is_ok(status)) {
    status = loomc_target_profile_create_amdgpu(
        state->target_environment, &profile_options, loomc_allocator_system(),
        &state->target_profile);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_compiler_create(state->context, NULL,
                                   loomc_allocator_system(), &state->compiler);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_pass_program_create_empty(
        state->context, NULL, loomc_allocator_system(), &state->pass_program);
  }
  return status;
}

static loomc_status_t run_sample(benchmark_state_t* state,
                                 const char* function_symbol,
                                 const char* output_path, sample_t* sample) {
  memset(sample, 0, sizeof(*sample));
  loomc_module_t* module = NULL;
  loomc_result_t* result = NULL;
  loomc_status_t status = loomc_ok_status();
  const uint64_t total_start_ns = now_ns();

  uint64_t phase_start_ns = now_ns();
  status = loomc_module_deserialize_bytecode_from_source(
      state->context, state->workspace, state->source, NULL,
      loomc_allocator_system(), &module, &result);
  const uint64_t deserialize_end_ns = now_ns();
  sample->deserialize_us = (double)(deserialize_end_ns - phase_start_ns) / 1e3;
  if (!loomc_status_is_ok(status)) {
    goto cleanup;
  }
  if (!require_successful_result(result, "bytecode deserialization")) {
    status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                               "bytecode deserialization failed");
    goto cleanup;
  }
  loomc_result_release(result);
  result = NULL;

  const loomc_target_specialization_t specialization = {
      .function_symbol = loomc_make_cstring_view(function_symbol),
      .target_profile = state->target_profile,
  };
  loomc_target_specialization_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      .structure_size = sizeof(target_options),
      .specializations = &specialization,
      .specialization_count = 1,
  };
  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("loom_blas_jit_benchmark"),
  };
  phase_start_ns = now_ns();
  status = loomc_compile_module(state->compiler, state->workspace,
                                state->pass_program, module, &compile_options,
                                loomc_allocator_system(), &result);
  const uint64_t compile_end_ns = now_ns();
  sample->compile_us = (double)(compile_end_ns - phase_start_ns) / 1e3;
  if (!loomc_status_is_ok(status)) {
    goto cleanup;
  }
  if (!require_successful_result(result, "prepared-Low compilation")) {
    status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                               "prepared-Low compilation failed");
    goto cleanup;
  }
  loomc_result_release(result);
  result = NULL;

  loomc_amdgpu_emit_options_t amdgpu_options = {
      .type = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      .structure_size = sizeof(amdgpu_options),
  };
  loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = &amdgpu_options,
      .artifact_format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      .identifier = loomc_make_cstring_view("loom_blas.hsaco"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  phase_start_ns = now_ns();
  status = loomc_emit_module(state->target_environment, state->workspace,
                             module, &emit_options, loomc_allocator_system(),
                             &result);
  const uint64_t emit_end_ns = now_ns();
  sample->emit_us = (double)(emit_end_ns - phase_start_ns) / 1e3;
  if (!loomc_status_is_ok(status)) {
    goto cleanup;
  }
  if (!require_successful_result(result, "AMDGPU HSACO emission")) {
    status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                               "AMDGPU HSACO emission failed");
    goto cleanup;
  }
  const loomc_artifact_t* artifact = find_hsaco(result);
  if (artifact == NULL) {
    fprintf(stderr, "AMDGPU emission produced no HSACO artifact\n");
    status = loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                               "HSACO artifact not found");
    goto cleanup;
  }
  sample->artifact_bytes = loomc_byte_sequence_length(artifact->contents);
  sample->total_us = (double)(now_ns() - total_start_ns) / 1e3;
  if (output_path != NULL) {
    status = loomc_artifact_write_to_path(
        artifact, loomc_make_cstring_view(output_path),
        loomc_allocator_system());
  }

cleanup:
  if (!loomc_status_is_ok(status)) print_status(status);
  loomc_result_release(result);
  loomc_module_release(module);
  return status;
}

static int compare_double(const void* lhs, const void* rhs) {
  const double a = *(const double*)lhs;
  const double b = *(const double*)rhs;
  return (a > b) - (a < b);
}

static void print_distribution(const char* name, const double* values,
                               size_t count) {
  double* sorted = (double*)malloc(count * sizeof(*sorted));
  if (sorted == NULL) {
    fprintf(stderr, "failed to allocate statistics buffer\n");
    exit(1);
  }
  memcpy(sorted, values, count * sizeof(*sorted));
  qsort(sorted, count, sizeof(*sorted), compare_double);
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) sum += sorted[i];
  const size_t median_index = count / 2;
  const double median = count % 2 == 0
                            ? (sorted[median_index - 1] + sorted[median_index]) /
                                  2.0
                            : sorted[median_index];
  size_t p95_index = (count * 95 + 99) / 100;
  if (p95_index == 0) p95_index = 1;
  if (p95_index > count) p95_index = count;
  printf("\"%s\":{\"min_us\":%.3f,\"median_us\":%.3f,"
         "\"mean_us\":%.3f,\"p95_us\":%.3f,\"max_us\":%.3f}",
         name, sorted[0], median, sum / (double)count,
         sorted[p95_index - 1], sorted[count - 1]);
  free(sorted);
}

static void print_usage(FILE* file) {
  fprintf(file,
          "Usage: loomc-jit-benchmark <kernel.loombc> <target> <symbol> "
          "[iterations] [output.hsaco]\n");
}

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    print_usage(stderr);
    return 64;
  }
  const char* source_path = argv[1];
  const char* target = argv[2];
  const char* function_symbol = argv[3];
  const size_t iteration_count = argc >= 5 ? strtoull(argv[4], NULL, 10) : 100;
  const char* output_path = argc >= 6 ? argv[5] : NULL;
  if (iteration_count == 0) {
    fprintf(stderr, "iterations must be positive\n");
    return 64;
  }

  struct stat source_stat;
  if (stat(source_path, &source_stat) != 0) {
    perror(source_path);
    return 1;
  }

  benchmark_state_t state;
  const uint64_t setup_start_ns = now_ns();
  loomc_status_t status =
      benchmark_state_initialize(&state, source_path, target);
  const double setup_us = (double)(now_ns() - setup_start_ns) / 1e3;
  if (!loomc_status_is_ok(status)) {
    print_status(status);
    loomc_status_free(status);
    benchmark_state_deinitialize(&state);
    return 1;
  }

  sample_t warmup;
  status = run_sample(&state, function_symbol, NULL, &warmup);
  if (!loomc_status_is_ok(status)) {
    loomc_status_free(status);
    benchmark_state_deinitialize(&state);
    return 1;
  }

  sample_t* samples = (sample_t*)calloc(iteration_count, sizeof(*samples));
  double* deserialize = (double*)malloc(iteration_count * sizeof(double));
  double* compile = (double*)malloc(iteration_count * sizeof(double));
  double* emit = (double*)malloc(iteration_count * sizeof(double));
  double* total = (double*)malloc(iteration_count * sizeof(double));
  if (samples == NULL || deserialize == NULL || compile == NULL ||
      emit == NULL || total == NULL) {
    fprintf(stderr, "failed to allocate sample buffers\n");
    return 1;
  }
  for (size_t i = 0; i < iteration_count; ++i) {
    const char* current_output = i + 1 == iteration_count ? output_path : NULL;
    status = run_sample(&state, function_symbol, current_output, &samples[i]);
    if (!loomc_status_is_ok(status)) {
      loomc_status_free(status);
      benchmark_state_deinitialize(&state);
      return 1;
    }
    deserialize[i] = samples[i].deserialize_us;
    compile[i] = samples[i].compile_us;
    emit[i] = samples[i].emit_us;
    total[i] = samples[i].total_us;
  }

  printf("{\"schema\":\"loom-blas.loomc-jit-benchmark.v1\","
         "\"source\":\"%s\",\"target\":\"%s\",\"symbol\":\"%s\","
         "\"loombc_bytes\":%" PRIu64 ",\"hsaco_bytes\":%" PRIu64 ","
         "\"iterations\":%zu,\"warmup_iterations\":1,"
         "\"setup_us\":%.3f,\"phases\":{",
         source_path, target, function_symbol, (uint64_t)source_stat.st_size,
         samples[iteration_count - 1].artifact_bytes, iteration_count,
         setup_us);
  print_distribution("deserialize", deserialize, iteration_count);
  printf(",");
  print_distribution("compile", compile, iteration_count);
  printf(",");
  print_distribution("emit", emit, iteration_count);
  printf(",");
  print_distribution("total", total, iteration_count);
  printf("}}\n");

  free(total);
  free(emit);
  free(compile);
  free(deserialize);
  free(samples);
  benchmark_state_deinitialize(&state);
  return 0;
}
