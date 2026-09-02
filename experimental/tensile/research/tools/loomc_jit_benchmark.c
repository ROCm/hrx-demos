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
  loomc_source_t* config_source;
  loomc_module_t* config_module;
  loomc_link_index_t* link_index;
  loomc_linker_t* linker;
  loomc_target_profile_t* target_profile;
  loomc_compiler_t* compiler;
  loomc_pass_program_t* pass_program;
} benchmark_state_t;

typedef struct sample_t {
  double link_us;
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
  loomc_linker_release(state->linker);
  loomc_link_index_release(state->link_index);
  loomc_target_profile_release(state->target_profile);
  loomc_module_release(state->config_module);
  loomc_source_release(state->config_source);
  loomc_source_release(state->source);
  loomc_workspace_release(state->workspace);
  loomc_context_release(state->context);
  loomc_target_environment_release(state->target_environment);
  memset(state, 0, sizeof(*state));
}

static loomc_status_t benchmark_state_initialize(benchmark_state_t* state,
                                                 const char* source_path,
                                                 const char* config_path,
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
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create_from_path(
        loomc_make_cstring_view(config_path), &source_options,
        loomc_allocator_system(), &state->config_source);
  }
  if (loomc_status_is_ok(status)) {
    loomc_result_t* config_result = NULL;
    status = loomc_module_deserialize_bytecode_from_source(
        state->context, state->workspace, state->config_source, NULL,
        loomc_allocator_system(), &state->config_module, &config_result);
    if (loomc_status_is_ok(status) &&
        !require_successful_result(config_result,
                                   "config bytecode deserialization")) {
      status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                 "config bytecode deserialization failed");
    }
    loomc_result_release(config_result);
  }
  if (loomc_status_is_ok(status)) {
    loomc_link_index_builder_t* builder = NULL;
    loomc_result_t* index_result = NULL;
    status = loomc_link_index_builder_create(
        state->context, NULL, loomc_allocator_system(), &builder);
    const loomc_link_index_source_options_t source_index_options = {
        .provider_name = loomc_make_cstring_view("loom-blas-bytecode"),
        .role = LOOMC_LINK_PROVIDER_ROLE_INPUT,
    };
    if (loomc_status_is_ok(status)) {
      status = loomc_link_index_builder_add_source(
          builder, state->source, &source_index_options, NULL);
    }
    if (loomc_status_is_ok(status)) {
      status = loomc_link_index_builder_finish(
          builder, &state->link_index, &index_result);
    }
    if (loomc_status_is_ok(status) &&
        !require_successful_result(index_result, "bytecode indexing")) {
      status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                 "bytecode indexing failed");
    }
    loomc_result_release(index_result);
    loomc_link_index_builder_release(builder);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_linker_create(state->context, NULL,
                                 loomc_allocator_system(), &state->linker);
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
    loomc_target_pipeline_options_t pipeline_options = {
        .type = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
        .structure_size = sizeof(pipeline_options),
        .identifier = loomc_make_cstring_view("loom-blas-prepared-low"),
        .kind = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
        .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
        .source_to_low_max_errors = 20,
    };
    loomc_result_t* pipeline_result = NULL;
    status = loomc_pass_program_create_from_target_pipeline(
        state->context, &pipeline_options, loomc_allocator_system(),
        &state->pass_program, &pipeline_result);
    if (loomc_status_is_ok(status) &&
        !require_successful_result(pipeline_result,
                                   "prepared-Low pipeline preparation")) {
      status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                                 "prepared-Low pipeline preparation failed");
    }
    loomc_result_release(pipeline_result);
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
  const loomc_string_view_t root = loomc_make_cstring_view(function_symbol);
  loomc_link_options_t link_options = {
      .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
      .structure_size = sizeof(link_options),
      .next = &target_options,
      .link_index = state->link_index,
      .module_name = loomc_make_cstring_view("loom_blas_jit_benchmark"),
      .mode = LOOMC_LINK_MODE_LINK,
      .root_symbols = &root,
      .root_symbol_count = 1,
  };
  uint64_t phase_start_ns = now_ns();
  status = loomc_link_module(state->linker, state->workspace, &link_options,
                             &module, &result);
  const uint64_t link_end_ns = now_ns();
  sample->link_us = (double)(link_end_ns - phase_start_ns) / 1e3;
  if (!loomc_status_is_ok(status)) {
    goto cleanup;
  }
  if (!require_successful_result(result, "bytecode link")) {
    status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                               "bytecode link failed");
    goto cleanup;
  }
  loomc_result_release(result);
  result = NULL;

  loomc_compile_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = &target_options,
      .module_name = loomc_make_cstring_view("loom_blas_jit_benchmark"),
      .config_flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED,
      .config_module = state->config_module,
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
  if (!require_successful_result(result, "default pipeline compilation")) {
    status = loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                               "default pipeline compilation failed");
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
          "Usage: loomc-jit-benchmark <kernel.loombc> <config.loombc> "
          "<target> <symbol> "
          "[iterations] [output.hsaco]\n");
}

int main(int argc, char** argv) {
  if (argc < 5 || argc > 7) {
    print_usage(stderr);
    return 64;
  }
  const char* source_path = argv[1];
  const char* config_path = argv[2];
  const char* target = argv[3];
  const char* function_symbol = argv[4];
  const size_t iteration_count = argc >= 6 ? strtoull(argv[5], NULL, 10) : 100;
  const char* output_path = argc >= 7 ? argv[6] : NULL;
  if (iteration_count == 0) {
    fprintf(stderr, "iterations must be positive\n");
    return 64;
  }

  struct stat source_stat;
  if (stat(source_path, &source_stat) != 0) {
    perror(source_path);
    return 1;
  }
  struct stat config_stat;
  if (stat(config_path, &config_stat) != 0) {
    perror(config_path);
    return 1;
  }

  benchmark_state_t state;
  const uint64_t setup_start_ns = now_ns();
  loomc_status_t status =
      benchmark_state_initialize(&state, source_path, config_path, target);
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
  double* link = (double*)malloc(iteration_count * sizeof(double));
  double* compile = (double*)malloc(iteration_count * sizeof(double));
  double* emit = (double*)malloc(iteration_count * sizeof(double));
  double* total = (double*)malloc(iteration_count * sizeof(double));
  if (samples == NULL || link == NULL || compile == NULL ||
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
    link[i] = samples[i].link_us;
    compile[i] = samples[i].compile_us;
    emit[i] = samples[i].emit_us;
    total[i] = samples[i].total_us;
  }

  printf("{\"schema\":\"loom-blas.loomc-jit-benchmark.v1\","
         "\"source\":\"%s\",\"config\":\"%s\",\"target\":\"%s\","
         "\"symbol\":\"%s\",\"loombc_bytes\":%" PRIu64 ","
         "\"config_loombc_bytes\":%" PRIu64 ",\"hsaco_bytes\":%" PRIu64 ","
         "\"iterations\":%zu,\"warmup_iterations\":1,"
         "\"setup_us\":%.3f,\"phases\":{",
         source_path, config_path, target, function_symbol,
         (uint64_t)source_stat.st_size, (uint64_t)config_stat.st_size,
         samples[iteration_count - 1].artifact_bytes, iteration_count, setup_us);
  print_distribution("link", link, iteration_count);
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
  free(link);
  free(samples);
  benchmark_state_deinitialize(&state);
  return 0;
}
