#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
  std::string backend = "hipblaslt";
  std::string type = "f16";
  std::string device_name;
  std::string device_arch;
  int device = 0;
  int64_t m = 256;
  int64_t n = 256;
  int64_t k = 256;
  int warmup = 1;
  int iterations = 5;
  size_t reference_samples = 0;
  size_t diagnostic_mismatches = 0;
  int max_algorithms = 0;
  std::optional<int> solution_index;
  size_t workspace_bytes = 32U * 1024U * 1024U;
  std::filesystem::path hsaco;
  std::string kernel;
  int tile_m = 16;
  int tile_n = 16;
  int workgroup_x = 32;
  int workgroup_y = 1;
  int workgroup_z = 1;
  std::optional<int> grid_x;
  std::optional<int> grid_y;
  std::filesystem::path dump_output;
  bool flatten_grid = false;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check_hip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    fail(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

void check_lt(hipblasStatus_t status, std::string_view operation) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    fail(std::string(operation) + ": hipBLASLt status " +
         std::to_string(static_cast<int>(status)));
  }
}

void check_rocblas(rocblas_status status, std::string_view operation) {
  if (status != rocblas_status_success) {
    fail(std::string(operation) + ": rocBLAS status " +
         std::to_string(static_cast<int>(status)));
  }
}

std::string json_escape(std::string_view input) {
  std::ostringstream out;
  for (const unsigned char c : input) {
    switch (c) {
    case '"': out << "\\\""; break;
    case '\\': out << "\\\\"; break;
    case '\b': out << "\\b"; break;
    case '\f': out << "\\f"; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default:
      if (c < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(c) << std::dec;
      } else {
        out << c;
      }
    }
  }
  return out.str();
}

int parse_int(const char *text, std::string_view option) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    fail("invalid value for " + std::string(option) + ": " + text);
  }
  return static_cast<int>(value);
}

int parse_signed_int(const char *text, std::string_view option) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    fail("invalid value for " + std::string(option) + ": " + text);
  }
  return static_cast<int>(value);
}

int64_t parse_i64(const char *text, std::string_view option) {
  char *end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0) {
    fail("invalid value for " + std::string(option) + ": " + text);
  }
  return static_cast<int64_t>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto next = [&]() -> const char * {
      if (++i >= argc) fail("missing value for " + std::string(arg));
      return argv[i];
    };
    if (arg == "--backend") options.backend = next();
    else if (arg == "--type") options.type = next();
    else if (arg == "--device") options.device = parse_int(next(), arg);
    else if (arg == "--m") options.m = parse_i64(next(), arg);
    else if (arg == "--n") options.n = parse_i64(next(), arg);
    else if (arg == "--k") options.k = parse_i64(next(), arg);
    else if (arg == "--warmup") options.warmup = parse_int(next(), arg);
    else if (arg == "--iterations") options.iterations = parse_int(next(), arg);
    else if (arg == "--reference-samples")
      options.reference_samples = static_cast<size_t>(parse_int(next(), arg));
    else if (arg == "--diagnostic-mismatches")
      options.diagnostic_mismatches =
          static_cast<size_t>(parse_int(next(), arg));
    else if (arg == "--max-algorithms")
      options.max_algorithms = parse_int(next(), arg);
    else if (arg == "--solution-index")
      options.solution_index = parse_signed_int(next(), arg);
    else if (arg == "--workspace-mib")
      options.workspace_bytes = static_cast<size_t>(parse_int(next(), arg)) *
                                1024U * 1024U;
    else if (arg == "--hsaco") options.hsaco = next();
    else if (arg == "--kernel") options.kernel = next();
    else if (arg == "--tile-m") options.tile_m = parse_int(next(), arg);
    else if (arg == "--tile-n") options.tile_n = parse_int(next(), arg);
    else if (arg == "--workgroup-size" || arg == "--workgroup-x")
      options.workgroup_x = parse_int(next(), arg);
    else if (arg == "--workgroup-y")
      options.workgroup_y = parse_int(next(), arg);
    else if (arg == "--workgroup-z")
      options.workgroup_z = parse_int(next(), arg);
    else if (arg == "--grid-x") options.grid_x = parse_int(next(), arg);
    else if (arg == "--grid-y") options.grid_y = parse_int(next(), arg);
    else if (arg == "--dump-output") options.dump_output = next();
    else if (arg == "--flatten-grid")
      options.flatten_grid = true;
    else if (arg == "--help") {
      std::cout << "Usage: blas-probe [--backend hipblaslt|rocblas|loom|compare] "
                   "[--type f16|bf16] "
                   "[--device N] [--m N --n N --k N] [--warmup N] "
                   "[--iterations N] [--max-algorithms N] "
                   "[--reference-samples N] "
                   "[--diagnostic-mismatches N] "
                   "[--solution-index N] "
                   "[--workspace-mib N] [--hsaco FILE --kernel SYMBOL] "
                   "[--tile-m N --tile-n N] "
                   "[--workgroup-x N --workgroup-y N --workgroup-z N] "
                   "[--grid-x N --grid-y N] "
                   "[--dump-output FILE] "
                   "[--flatten-grid]\n";
      std::exit(0);
    } else {
      fail("unknown option: " + std::string(arg));
    }
  }
  if (options.backend != "hipblaslt" && options.backend != "rocblas" &&
      options.backend != "loom" && options.backend != "compare")
    fail("--backend must be hipblaslt, rocblas, loom, or compare");
  if (options.type != "f16" && options.type != "bf16")
    fail("--type must be f16 or bf16");
  if (options.iterations == 0) fail("--iterations must be positive");
  if (options.backend == "loom" || options.backend == "compare") {
    if (options.hsaco.empty() || options.kernel.empty())
      fail("--backend loom requires --hsaco and --kernel");
    if (!std::filesystem::is_regular_file(options.hsaco) ||
        std::filesystem::file_size(options.hsaco) == 0)
      fail("--hsaco must name a nonempty file");
    if (options.tile_m == 0 || options.tile_n == 0 ||
        options.workgroup_x == 0 || options.workgroup_y == 0 ||
        options.workgroup_z == 0)
      fail("Loom launch dimensions must be positive");
  }
  return options;
}

struct DeviceBuffer {
  void *pointer = nullptr;
  explicit DeviceBuffer(size_t bytes) {
    if (bytes != 0) check_hip(hipMalloc(&pointer, bytes), "hipMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (pointer != nullptr) (void)hipFree(pointer);
  }
};

struct Event {
  hipEvent_t value = nullptr;
  Event() { check_hip(hipEventCreate(&value), "hipEventCreate"); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  ~Event() {
    if (value != nullptr) (void)hipEventDestroy(value);
  }
};

struct Module {
  hipModule_t value = nullptr;
  explicit Module(const std::filesystem::path &path) {
    check_hip(hipModuleLoad(&value, path.c_str()), "hipModuleLoad");
  }
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  ~Module() {
    if (value != nullptr) (void)hipModuleUnload(value);
  }
};

struct Inputs {
  std::vector<uint16_t> a;
  std::vector<uint16_t> b;
  std::vector<uint16_t> c;
  std::vector<uint16_t> expected;
  std::vector<size_t> expected_indices;
};

uint16_t element_from_float(float value, std::string_view type) {
  uint16_t bits = 0;
  if (type == "f16") {
    const __half converted = __float2half(value);
    static_assert(sizeof(converted) == sizeof(bits));
    std::memcpy(&bits, &converted, sizeof(bits));
  } else {
    uint32_t float_bits = 0;
    static_assert(sizeof(float_bits) == sizeof(value));
    std::memcpy(&float_bits, &value, sizeof(float_bits));
    const uint32_t round_to_nearest_even =
        0x7fffU + ((float_bits >> 16U) & 1U);
    bits = static_cast<uint16_t>((float_bits + round_to_nearest_even) >> 16U);
  }
  return bits;
}

float element_to_float(uint16_t bits, std::string_view type) {
  if (type == "f16") {
    __half value;
    std::memcpy(&value, &bits, sizeof(bits));
    return __half2float(value);
  }
  const uint32_t float_bits = static_cast<uint32_t>(bits) << 16U;
  float value = 0.0F;
  static_assert(sizeof(float_bits) == sizeof(value));
  std::memcpy(&value, &float_bits, sizeof(value));
  return value;
}

float value_for(size_t index, int salt) {
  const int value = static_cast<int>((index * 17U + static_cast<size_t>(salt)) % 23U) - 11;
  return static_cast<float>(value) / 32.0F;
}

Inputs make_inputs(const Options &options) {
  Inputs result;
  result.a.resize(static_cast<size_t>(options.m * options.k));
  result.b.resize(static_cast<size_t>(options.k * options.n));
  result.c.resize(static_cast<size_t>(options.m * options.n));
  for (size_t i = 0; i < result.a.size(); ++i)
    result.a[i] = element_from_float(value_for(i, 3), options.type);
  for (size_t i = 0; i < result.b.size(); ++i)
    result.b[i] = element_from_float(value_for(i, 7), options.type);
  for (size_t i = 0; i < result.c.size(); ++i)
    result.c[i] = element_from_float(value_for(i, 11), options.type);

  const size_t output_elements = result.c.size();
  const size_t sample_count =
      options.reference_samples == 0
          ? output_elements
          : std::min(options.reference_samples, output_elements);
  result.expected.reserve(sample_count);
  result.expected_indices.reserve(sample_count);
  for (size_t sample = 0; sample < sample_count; ++sample) {
    // A full reference visits elements in storage order. The sampled reference
    // uses an odd stride over the output ring so skinny matrices cover both
    // axes without allocating or computing the entire CPU GEMM.
    const size_t output_index = options.reference_samples == 0
                                    ? sample
                                    : (sample * 104729U) % output_elements;
    const int64_t row = static_cast<int64_t>(output_index % options.m);
    const int64_t column = static_cast<int64_t>(output_index / options.m);
    float sum = 0.0F;
    for (int64_t inner = 0; inner < options.k; ++inner) {
      const float a = element_to_float(
          result.a[static_cast<size_t>(row + inner * options.m)], options.type);
      const float b = element_to_float(
          result.b[static_cast<size_t>(inner + column * options.k)], options.type);
      sum = std::fma(a, b, sum);
    }
    result.expected_indices.push_back(output_index);
    result.expected.push_back(element_from_float(sum, options.type));
  }
  return result;
}

struct ErrorSummary {
  double max_absolute = 0.0;
  double max_relative = 0.0;
  size_t mismatches = 0;
};

ErrorSummary compare(const std::vector<uint16_t> &actual,
                     const std::vector<uint16_t> &expected,
                     const std::vector<size_t> &expected_indices,
                     size_t diagnostic_mismatches, std::string_view type) {
  ErrorSummary summary;
  size_t diagnosed = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double got = element_to_float(actual[expected_indices[i]], type);
    const double want = element_to_float(expected[i], type);
    const double absolute = std::abs(got - want);
    const double relative = absolute / std::max(std::abs(want), 1.0e-3);
    summary.max_absolute = std::max(summary.max_absolute, absolute);
    summary.max_relative = std::max(summary.max_relative, relative);
    // The inputs are exact binary fractions; tolerate accumulation-order and
    // final FP16 rounding differences without accepting gross failures.
    if (absolute > 0.02 + 0.01 * std::abs(want)) {
      ++summary.mismatches;
      if (diagnosed < diagnostic_mismatches) {
        std::cerr << "mismatch index=" << expected_indices[i]
                  << " got=" << got << " want=" << want
                  << " absolute=" << absolute << '\n';
        ++diagnosed;
      }
    }
  }
  return summary;
}

template <typename Launch>
std::vector<float> measure(hipStream_t stream, int warmup, int iterations,
                           Launch launch) {
  for (int i = 0; i < warmup; ++i) launch();
  check_hip(hipStreamSynchronize(stream), "warmup synchronize");
  Event start;
  Event stop;
  std::vector<float> samples;
  samples.reserve(static_cast<size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    check_hip(hipEventRecord(start.value, stream), "record start");
    launch();
    check_hip(hipEventRecord(stop.value, stream), "record stop");
    check_hip(hipEventSynchronize(stop.value), "synchronize stop");
    float milliseconds = 0.0F;
    check_hip(hipEventElapsedTime(&milliseconds, start.value, stop.value),
              "elapsed time");
    samples.push_back(milliseconds * 1000.0F);
  }
  return samples;
}

void print_samples(const std::vector<float> &samples) {
  std::cout << '[';
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i != 0) std::cout << ',';
    std::cout << std::fixed << std::setprecision(3) << samples[i];
  }
  std::cout << ']';
}

struct CommonState {
  Inputs inputs;
  DeviceBuffer a;
  DeviceBuffer b;
  DeviceBuffer c;
  DeviceBuffer d;
  DeviceBuffer workspace;
  hipStream_t stream = nullptr;
  size_t diagnostic_mismatches = 0;

  explicit CommonState(const Options &options)
      : inputs(make_inputs(options)), a(inputs.a.size() * sizeof(uint16_t)),
        b(inputs.b.size() * sizeof(uint16_t)), c(inputs.c.size() * sizeof(uint16_t)),
        d(inputs.c.size() * sizeof(uint16_t)), workspace(options.workspace_bytes),
        diagnostic_mismatches(options.diagnostic_mismatches), type(options.type) {
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");
    check_hip(hipMemcpy(a.pointer, inputs.a.data(), inputs.a.size() * sizeof(uint16_t),
                       hipMemcpyHostToDevice), "copy A");
    check_hip(hipMemcpy(b.pointer, inputs.b.data(), inputs.b.size() * sizeof(uint16_t),
                       hipMemcpyHostToDevice), "copy B");
    check_hip(hipMemcpy(c.pointer, inputs.c.data(), inputs.c.size() * sizeof(uint16_t),
                       hipMemcpyHostToDevice), "copy C");
  }

  CommonState(const CommonState &) = delete;
  CommonState &operator=(const CommonState &) = delete;

  ~CommonState() {
    if (stream != nullptr) (void)hipStreamDestroy(stream);
  }

  ErrorSummary read_and_compare() {
    const std::vector<uint16_t> result = read_result();
    return compare(result, inputs.expected, inputs.expected_indices,
                   diagnostic_mismatches, type);
  }

  std::vector<uint16_t> read_result() {
    std::vector<uint16_t> result(inputs.c.size());
    check_hip(hipMemcpy(result.data(), d.pointer, result.size() * sizeof(uint16_t),
                        hipMemcpyDeviceToHost), "copy D");
    return result;
  }

  void dump_result(const std::filesystem::path &path) {
    const std::vector<uint16_t> result = read_result();
    std::ofstream output(path, std::ios::binary);
    if (!output) fail("cannot open --dump-output: " + path.string());
    output.write(reinterpret_cast<const char *>(result.data()),
                 static_cast<std::streamsize>(result.size() * sizeof(uint16_t)));
    if (!output) fail("cannot write --dump-output: " + path.string());
  }

  std::string type;
};

struct LtLayouts {
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulDesc_t operation = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;

  explicit LtLayouts(const Options &options) {
    const hipDataType storage_type =
        options.type == "f16" ? HIP_R_16F : HIP_R_16BF;
    check_lt(hipblasLtMatrixLayoutCreate(&a, storage_type, options.m, options.k,
                                         options.m), "create A layout");
    check_lt(hipblasLtMatrixLayoutCreate(&b, storage_type, options.k, options.n,
                                         options.k), "create B layout");
    check_lt(hipblasLtMatrixLayoutCreate(&c, storage_type, options.m, options.n,
                                         options.m), "create C layout");
    check_lt(hipblasLtMatrixLayoutCreate(&d, storage_type, options.m, options.n,
                                         options.m), "create D layout");
    check_lt(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F,
                                      HIP_R_32F), "create operation");
    const hipblasOperation_t no_transpose = HIPBLAS_OP_N;
    check_lt(hipblasLtMatmulDescSetAttribute(operation,
                                             HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &no_transpose,
                                             sizeof(no_transpose)), "set transA");
    check_lt(hipblasLtMatmulDescSetAttribute(operation,
                                             HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &no_transpose,
                                             sizeof(no_transpose)), "set transB");
    check_lt(hipblasLtMatmulPreferenceCreate(&preference), "create preference");
    check_lt(hipblasLtMatmulPreferenceSetAttribute(
                 preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                 &options.workspace_bytes, sizeof(options.workspace_bytes)),
             "set workspace preference");
  }

  ~LtLayouts() {
    if (preference != nullptr) hipblasLtMatmulPreferenceDestroy(preference);
    if (operation != nullptr) hipblasLtMatmulDescDestroy(operation);
    if (a != nullptr) hipblasLtMatrixLayoutDestroy(a);
    if (b != nullptr) hipblasLtMatrixLayoutDestroy(b);
    if (c != nullptr) hipblasLtMatrixLayoutDestroy(c);
    if (d != nullptr) hipblasLtMatrixLayoutDestroy(d);
  }
};

void run_hipblaslt(const Options &options, CommonState &state) {
  hipblasLtHandle_t handle = nullptr;
  check_lt(hipblasLtCreate(&handle), "hipblasLtCreate");
  LtLayouts layouts(options);
  const float alpha = 1.0F;
  const float beta = 0.0F;

  std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
  const hipDataType storage_type =
      options.type == "f16" ? HIP_R_16F : HIP_R_16BF;
  check_lt(hipblaslt_ext::getAllAlgos(
               handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
               HIPBLAS_OP_N, HIPBLAS_OP_N, storage_type, storage_type,
               storage_type, storage_type, HIPBLAS_COMPUTE_32F, algorithms),
           "getAllAlgos");

  hipblasLtMatmulHeuristicResult_t default_result{};
  int default_count = 0;
  const hipblasStatus_t default_status = hipblasLtMatmulAlgoGetHeuristic(
      handle, layouts.operation, layouts.a, layouts.b, layouts.c, layouts.d,
      layouts.preference, 1, &default_result, &default_count);

  std::cout << "{\n  \"schema\":\"loom-blas.probe.v1\",\n"
            << "  \"backend\":\"hipblaslt\",\n"
            << "  \"device\":{\"ordinal\":" << options.device
            << ",\"name\":\"" << json_escape(options.device_name)
            << "\",\"arch\":\"" << json_escape(options.device_arch) << "\"},\n"
            << "  \"request\":{\"m\":" << options.m << ",\"n\":" << options.n
            << ",\"k\":" << options.k
            << ",\"trans_a\":\"N\",\"trans_b\":\"N\",\"a_type\":\""
            << options.type << "\",\"b_type\":\"" << options.type
            << "\",\"c_type\":\"" << options.type << "\",\"d_type\":\""
            << options.type << "\","
               "\"compute_type\":\"f32\",\"alpha\":1.0,\"beta\":0.0},\n"
            << "  \"enumerated_count\":" << algorithms.size() << ",\n"
            << "  \"default_query\":{\"status\":"
            << static_cast<int>(default_status) << ",\"count\":" << default_count;
  if (default_count > 0) {
    auto algorithm = default_result.algo;
    std::cout << ",\"index\":" << hipblaslt_ext::getIndexFromAlgo(algorithm)
              << ",\"workspace_bytes\":" << default_result.workspaceSize;
    size_t default_workspace = 0;
    const hipblasStatus_t support = hipblaslt_ext::matmulIsAlgoSupported(
        handle, layouts.operation, &alpha, layouts.a, layouts.b, &beta,
        layouts.c, layouts.d, algorithm, default_workspace);
    if (support == HIPBLAS_STATUS_SUCCESS &&
        default_workspace <= options.workspace_bytes) {
      auto launch = [&]() {
        check_lt(hipblasLtMatmul(handle, layouts.operation, &alpha,
                                state.a.pointer, layouts.a, state.b.pointer,
                                layouts.b, &beta, state.c.pointer, layouts.c,
                                state.d.pointer, layouts.d, &algorithm,
                                state.workspace.pointer, default_workspace,
                                state.stream), "hipblasLtMatmul(default)");
      };
      const std::vector<float> samples = measure(
          state.stream, options.warmup, options.iterations, launch);
      const ErrorSummary error = state.read_and_compare();
      std::cout << ",\"time_us\":";
      print_samples(samples);
      std::cout << ",\"correctness\":{\"mismatches\":" << error.mismatches
                << ",\"max_absolute\":" << error.max_absolute
                << ",\"max_relative\":" << error.max_relative << "}";
    }
  }
  std::cout << "},\n  \"algorithms\":[\n";

  bool first = true;
  int tested = 0;
  for (auto &candidate : algorithms) {
    if (options.max_algorithms != 0 && tested >= options.max_algorithms) break;
    const int index = hipblaslt_ext::getIndexFromAlgo(candidate.algo);
    if (options.solution_index.has_value() &&
        index != options.solution_index.value())
      continue;
    size_t required_workspace = 0;
    const hipblasStatus_t support = hipblaslt_ext::matmulIsAlgoSupported(
        handle, layouts.operation, &alpha, layouts.a, layouts.b, &beta,
        layouts.c, layouts.d, candidate.algo, required_workspace);
    if (support != HIPBLAS_STATUS_SUCCESS ||
        required_workspace > options.workspace_bytes)
      continue;
    ++tested;
    auto algorithm = candidate.algo;
    const std::string solution =
        hipblaslt_ext::getSolutionNameFromAlgo(handle, algorithm);
    const std::string kernel =
        hipblaslt_ext::getKernelNameFromAlgo(handle, algorithm);
    auto launch = [&]() {
      check_lt(hipblasLtMatmul(handle, layouts.operation, &alpha, state.a.pointer,
                              layouts.a, state.b.pointer, layouts.b, &beta,
                              state.c.pointer, layouts.c, state.d.pointer,
                              layouts.d, &algorithm, state.workspace.pointer,
                              required_workspace, state.stream), "hipblasLtMatmul");
    };
    const std::vector<float> samples = measure(
        state.stream, options.warmup, options.iterations, launch);
    const ErrorSummary error = state.read_and_compare();
    if (!first) std::cout << ",\n";
    first = false;
    std::cout << "    {\"index\":" << index
              << ",\"solution_name\":\"" << json_escape(solution)
              << "\",\"kernel_name\":\"" << json_escape(kernel)
              << "\",\"workspace_bytes\":" << required_workspace
              << ",\"time_us\":";
    print_samples(samples);
    std::cout << ",\"correctness\":{\"mismatches\":" << error.mismatches
              << ",\"max_absolute\":" << error.max_absolute
              << ",\"max_relative\":" << error.max_relative << "}}";
  }
  std::cout << "\n  ]\n}\n";
  check_lt(hipblasLtDestroy(handle), "hipblasLtDestroy");
}

void run_rocblas(const Options &options, CommonState &state) {
  rocblas_handle handle = nullptr;
  check_rocblas(rocblas_create_handle(&handle), "rocblas_create_handle");
  check_rocblas(rocblas_set_stream(handle, state.stream), "rocblas_set_stream");
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const rocblas_int m = static_cast<rocblas_int>(options.m);
  const rocblas_int n = static_cast<rocblas_int>(options.n);
  const rocblas_int k = static_cast<rocblas_int>(options.k);
  const rocblas_int lda = m;
  const rocblas_int ldb = k;
  const rocblas_int ldc = m;
  const rocblas_int ldd = m;
  const rocblas_datatype storage_type = options.type == "f16"
                                            ? rocblas_datatype_f16_r
                                            : rocblas_datatype_bf16_r;

#define GEMM_ARGUMENTS                                                         \
  handle, rocblas_operation_none, rocblas_operation_none, m, n, k, &alpha,    \
      state.a.pointer, storage_type, lda, state.b.pointer, storage_type, ldb,  \
      &beta, state.c.pointer, storage_type, ldc, state.d.pointer, storage_type,\
      ldd, rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index
#define rocblas_gemm_ex_expanded(...) rocblas_gemm_ex(__VA_ARGS__)

  rocblas_int count = 0;
  check_rocblas(rocblas_gemm_ex_get_solutions(
                    GEMM_ARGUMENTS, rocblas_gemm_flags_none, nullptr, &count),
                "rocblas_gemm_ex_get_solutions(count)");
  std::vector<rocblas_int> solutions(static_cast<size_t>(count));
  check_rocblas(rocblas_gemm_ex_get_solutions(
                    GEMM_ARGUMENTS, rocblas_gemm_flags_none, solutions.data(),
                    &count),
                "rocblas_gemm_ex_get_solutions(values)");

  std::cout << "{\n  \"schema\":\"loom-blas.probe.v1\",\n"
            << "  \"backend\":\"rocblas\",\n"
            << "  \"device\":{\"ordinal\":" << options.device
            << ",\"name\":\"" << json_escape(options.device_name)
            << "\",\"arch\":\"" << json_escape(options.device_arch) << "\"},\n"
            << "  \"request\":{\"m\":" << options.m << ",\"n\":" << options.n
            << ",\"k\":" << options.k
            << ",\"trans_a\":\"N\",\"trans_b\":\"N\",\"a_type\":\""
            << options.type << "\",\"b_type\":\"" << options.type
            << "\",\"c_type\":\"" << options.type << "\",\"d_type\":\""
            << options.type << "\","
               "\"compute_type\":\"f32\",\"alpha\":1.0,\"beta\":0.0},\n"
            << "  \"enumerated_count\":" << count << ",\n";

  auto default_launch = [&]() {
    check_rocblas(rocblas_gemm_ex_expanded(GEMM_ARGUMENTS, 0,
                                           rocblas_gemm_flags_none),
                  "rocblas_gemm_ex(default)");
  };
  const std::vector<float> default_samples = measure(
      state.stream, options.warmup, options.iterations, default_launch);
  const ErrorSummary default_error = state.read_and_compare();
  std::cout << "  \"default_query\":{\"index\":0,\"time_us\":";
  print_samples(default_samples);
  std::cout << ",\"correctness\":{\"mismatches\":"
            << default_error.mismatches << ",\"max_absolute\":"
            << default_error.max_absolute << ",\"max_relative\":"
            << default_error.max_relative << "}},\n"
            << "  \"algorithms\":[\n";

  bool first = true;
  int tested = 0;
  for (const rocblas_int solution : solutions) {
    if (options.max_algorithms != 0 && tested >= options.max_algorithms) break;
    if (options.solution_index.has_value() &&
        solution != options.solution_index.value())
      continue;
    ++tested;
    auto launch = [&]() {
      check_rocblas(rocblas_gemm_ex_expanded(GEMM_ARGUMENTS, solution,
                                             rocblas_gemm_flags_none),
                    "rocblas_gemm_ex");
    };
    const std::vector<float> samples = measure(
        state.stream, options.warmup, options.iterations, launch);
    const ErrorSummary error = state.read_and_compare();
    if (!first) std::cout << ",\n";
    first = false;
    std::cout << "    {\"index\":" << solution << ",\"time_us\":";
    print_samples(samples);
    std::cout << ",\"correctness\":{\"mismatches\":" << error.mismatches
              << ",\"max_absolute\":" << error.max_absolute
              << ",\"max_relative\":" << error.max_relative << "}}";
  }
  std::cout << "\n  ]\n}\n";
#undef GEMM_ARGUMENTS
#undef rocblas_gemm_ex_expanded
  check_rocblas(rocblas_destroy_handle(handle), "rocblas_destroy_handle");
}

void run_loom(const Options &options, CommonState &state) {
  Module module(options.hsaco);
  hipFunction_t function = nullptr;
  check_hip(hipModuleGetFunction(&function, module.value, options.kernel.c_str()),
            "hipModuleGetFunction");
  unsigned grid_x = static_cast<unsigned>(
      (options.n + options.tile_n - 1) / options.tile_n);
  unsigned grid_y = static_cast<unsigned>(
      (options.m + options.tile_m - 1) / options.tile_m);
  if (options.grid_x.has_value())
    grid_x = static_cast<unsigned>(options.grid_x.value());
  if (options.grid_y.has_value())
    grid_y = static_cast<unsigned>(options.grid_y.value());
  if (options.flatten_grid) {
    grid_x *= grid_y;
    grid_y = 1;
  }
  const unsigned block_x = static_cast<unsigned>(options.workgroup_x);
  const unsigned block_y = static_cast<unsigned>(options.workgroup_y);
  const unsigned block_z = static_cast<unsigned>(options.workgroup_z);
  auto launch = [&]() {
    void *arguments[] = {&state.a.pointer, &state.b.pointer, &state.c.pointer,
                         &state.d.pointer};
    check_hip(hipModuleLaunchKernel(function, grid_x, grid_y, 1, block_x,
                                    block_y, block_z, 0, state.stream,
                                    arguments, nullptr),
              "hipModuleLaunchKernel");
  };
  const std::vector<float> samples =
      measure(state.stream, options.warmup, options.iterations, launch);
  const ErrorSummary error = state.read_and_compare();
  if (!options.dump_output.empty()) state.dump_result(options.dump_output);
  std::cout << "{\n  \"schema\":\"loom-blas.probe.v1\",\n"
            << "  \"backend\":\"loom\",\n"
            << "  \"device\":{\"ordinal\":" << options.device
            << ",\"name\":\"" << json_escape(options.device_name)
            << "\",\"arch\":\"" << json_escape(options.device_arch)
            << "\"},\n"
            << "  \"request\":{\"m\":" << options.m << ",\"n\":"
            << options.n << ",\"k\":" << options.k
            << ",\"trans_a\":\"N\",\"trans_b\":\"N\",\"a_type\":\""
            << options.type << "\",\"b_type\":\"" << options.type
            << "\",\"c_type\":\"" << options.type << "\",\"d_type\":\""
            << options.type << "\","
               "\"compute_type\":\"f32\",\"alpha\":1.0,\"beta\":0.0},\n"
            << "  \"artifact\":{\"hsaco\":\""
            << json_escape(options.hsaco.string()) << "\",\"kernel\":\""
            << json_escape(options.kernel) << "\"},\n"
            << "  \"launch\":{\"grid\":[" << grid_x << ',' << grid_y
            << ",1],\"workgroup\":[" << block_x << ',' << block_y << ','
            << block_z << "]},\n"
            << "  \"time_us\":";
  print_samples(samples);
  std::cout << ",\n  \"correctness\":{\"mismatches\":" << error.mismatches
            << ",\"max_absolute\":" << error.max_absolute
            << ",\"max_relative\":" << error.max_relative << "}\n}\n";
}

void run_compare(const Options &options, CommonState &state) {
  if (!options.solution_index.has_value())
    fail("--backend compare requires --solution-index");

  hipblasLtHandle_t handle = nullptr;
  check_lt(hipblasLtCreate(&handle), "hipblasLtCreate");
  LtLayouts layouts(options);
  const float alpha = 1.0F;
  const float beta = 0.0F;
  std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
  const hipDataType storage_type =
      options.type == "f16" ? HIP_R_16F : HIP_R_16BF;
  check_lt(hipblaslt_ext::getAllAlgos(
               handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
               HIPBLAS_OP_N, HIPBLAS_OP_N, storage_type, storage_type,
               storage_type, storage_type, HIPBLAS_COMPUTE_32F, algorithms),
           "getAllAlgos");
  auto selected = std::find_if(
      algorithms.begin(), algorithms.end(), [&](auto candidate) {
        return hipblaslt_ext::getIndexFromAlgo(candidate.algo) ==
               options.solution_index.value();
      });
  if (selected == algorithms.end()) fail("requested hipBLASLt solution is absent");
  size_t required_workspace = 0;
  check_lt(hipblaslt_ext::matmulIsAlgoSupported(
               handle, layouts.operation, &alpha, layouts.a, layouts.b, &beta,
               layouts.c, layouts.d, selected->algo, required_workspace),
           "matmulIsAlgoSupported");
  if (required_workspace > options.workspace_bytes)
    fail("requested hipBLASLt solution exceeds workspace limit");
  auto algorithm = selected->algo;

  Module module(options.hsaco);
  hipFunction_t function = nullptr;
  check_hip(hipModuleGetFunction(&function, module.value, options.kernel.c_str()),
            "hipModuleGetFunction");
  unsigned grid_x = static_cast<unsigned>(
      (options.n + options.tile_n - 1) / options.tile_n);
  unsigned grid_y = static_cast<unsigned>(
      (options.m + options.tile_m - 1) / options.tile_m);
  if (options.grid_x.has_value())
    grid_x = static_cast<unsigned>(options.grid_x.value());
  if (options.grid_y.has_value())
    grid_y = static_cast<unsigned>(options.grid_y.value());
  if (options.flatten_grid) {
    grid_x *= grid_y;
    grid_y = 1;
  }
  const unsigned block_x = static_cast<unsigned>(options.workgroup_x);
  const unsigned block_y = static_cast<unsigned>(options.workgroup_y);
  const unsigned block_z = static_cast<unsigned>(options.workgroup_z);
  auto launch_loom = [&]() {
    void *arguments[] = {&state.a.pointer, &state.b.pointer, &state.c.pointer,
                         &state.d.pointer};
    check_hip(hipModuleLaunchKernel(function, grid_x, grid_y, 1, block_x,
                                    block_y, block_z, 0, state.stream,
                                    arguments, nullptr),
              "hipModuleLaunchKernel");
  };
  auto launch_incumbent = [&]() {
    check_lt(hipblasLtMatmul(handle, layouts.operation, &alpha, state.a.pointer,
                            layouts.a, state.b.pointer, layouts.b, &beta,
                            state.c.pointer, layouts.c, state.d.pointer,
                            layouts.d, &algorithm, state.workspace.pointer,
                            required_workspace, state.stream),
             "hipblasLtMatmul(compare)");
  };

  for (int i = 0; i < options.warmup; ++i) {
    launch_loom();
    launch_incumbent();
  }
  check_hip(hipStreamSynchronize(state.stream), "compare warmup synchronize");
  launch_loom();
  check_hip(hipStreamSynchronize(state.stream), "Loom correctness synchronize");
  const ErrorSummary loom_error = state.read_and_compare();
  launch_incumbent();
  check_hip(hipStreamSynchronize(state.stream), "incumbent correctness synchronize");
  const ErrorSummary incumbent_error = state.read_and_compare();

  std::vector<float> loom_samples;
  std::vector<float> incumbent_samples;
  loom_samples.reserve(static_cast<size_t>(options.iterations));
  incumbent_samples.reserve(static_cast<size_t>(options.iterations));
  Event start;
  Event stop;
  auto timed = [&](auto &launch) {
    check_hip(hipEventRecord(start.value, state.stream), "record compare start");
    launch();
    check_hip(hipEventRecord(stop.value, state.stream), "record compare stop");
    check_hip(hipEventSynchronize(stop.value), "synchronize compare stop");
    float milliseconds = 0.0F;
    check_hip(hipEventElapsedTime(&milliseconds, start.value, stop.value),
              "compare elapsed time");
    return milliseconds * 1000.0F;
  };
  for (int i = 0; i < options.iterations; ++i) {
    if ((i & 1) == 0) {
      loom_samples.push_back(timed(launch_loom));
      incumbent_samples.push_back(timed(launch_incumbent));
    } else {
      incumbent_samples.push_back(timed(launch_incumbent));
      loom_samples.push_back(timed(launch_loom));
    }
  }

  std::cout << "{\n  \"schema\":\"loom-blas.paired-probe.v1\",\n"
            << "  \"backend\":\"compare\",\n"
            << "  \"device\":{\"ordinal\":" << options.device
            << ",\"name\":\"" << json_escape(options.device_name)
            << "\",\"arch\":\"" << json_escape(options.device_arch)
            << "\"},\n  \"request\":{\"m\":" << options.m << ",\"n\":"
            << options.n << ",\"k\":" << options.k
            << ",\"type\":\"" << options.type << "\"},\n"
            << "  \"loom\":{\"hsaco\":\"" << json_escape(options.hsaco.string())
            << "\",\"kernel\":\"" << json_escape(options.kernel)
            << "\",\"time_us\":";
  print_samples(loom_samples);
  std::cout << ",\"correctness\":{\"mismatches\":" << loom_error.mismatches
            << ",\"max_absolute\":" << loom_error.max_absolute << "}},\n"
            << "  \"incumbent\":{\"solution_index\":"
            << options.solution_index.value() << ",\"time_us\":";
  print_samples(incumbent_samples);
  std::cout << ",\"correctness\":{\"mismatches\":"
            << incumbent_error.mismatches << ",\"max_absolute\":"
            << incumbent_error.max_absolute << "}}\n}\n";
  check_lt(hipblasLtDestroy(handle), "hipblasLtDestroy");
}

} // namespace

int main(int argc, char **argv) {
  try {
    Options options = parse_options(argc, argv);
    check_hip(hipSetDevice(options.device), "hipSetDevice");
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, options.device),
              "hipGetDeviceProperties");
    options.device_name = properties.name;
    options.device_arch = properties.gcnArchName;
    std::cerr << "device=" << options.device << " name=" << properties.name
              << " arch=" << properties.gcnArchName << " backend="
              << options.backend << " shape=" << options.m << 'x' << options.n
              << 'x' << options.k << '\n';
    CommonState state(options);
    if (options.backend == "hipblaslt") run_hipblaslt(options, state);
    else if (options.backend == "rocblas") run_rocblas(options, state);
    else if (options.backend == "loom") run_loom(options, state);
    else run_compare(options, state);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
