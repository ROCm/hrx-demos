#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }
void hip_check(hipError_t status, std::string_view op) {
  if (status != hipSuccess) fail(std::string(op) + ": " + hipGetErrorString(status));
}
void lt_check(hipblasStatus_t status, std::string_view op) {
  if (status != HIPBLAS_STATUS_SUCCESS)
    fail(std::string(op) + ": status " + std::to_string(static_cast<int>(status)));
}
long long integer(const char *text, std::string_view option) {
  char *end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || value < 0) fail("invalid " + std::string(option));
  return value;
}
std::string escape(std::string_view text) {
  std::string result;
  for (char c : text) { if (c == '"' || c == '\\') result += '\\'; result += c; }
  return result;
}

struct Options {
  int device = 0;
  int64_t m = 1024, n = 1024, k = 1024;
  int warmup = 1, iterations = 2;
  size_t workspace = 32U << 20;
  std::string type = "bf16";
};
Options options(int argc, char **argv) {
  Options result;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto next = [&] { if (++i >= argc) fail("missing " + std::string(arg)); return argv[i]; };
    if (arg == "--device") result.device = static_cast<int>(integer(next(), arg));
    else if (arg == "--m") result.m = integer(next(), arg);
    else if (arg == "--n") result.n = integer(next(), arg);
    else if (arg == "--k") result.k = integer(next(), arg);
    else if (arg == "--warmup") result.warmup = static_cast<int>(integer(next(), arg));
    else if (arg == "--iterations") result.iterations = static_cast<int>(integer(next(), arg));
    else if (arg == "--workspace-mib") result.workspace = static_cast<size_t>(integer(next(), arg)) << 20;
    else if (arg == "--type") result.type = next();
    else if (arg == "--help") {
      std::cout << "Usage: hipblaslt-route-probe [--device N] [--m N --n N --k N] "
                   "[--type bf16|i8|fp8-fp8|fp8-bf8] [--warmup N --iterations N]\n";
      std::exit(0);
    } else fail("unknown option: " + std::string(arg));
  }
  if (!result.m || !result.n || !result.k || !result.iterations) fail("dimensions and iterations must be positive");
  return result;
}

struct Type {
  hipDataType a, b, c, d;
  hipblasComputeType_t compute;
  hipDataType scale;
  size_t a_size, b_size, c_size, d_size;
};
Type type_for(std::string_view name) {
  if (name == "bf16")
    return {HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
            HIPBLAS_COMPUTE_32F, HIP_R_32F, 2, 2, 2, 2};
  if (name == "i8")
    return {HIP_R_8I, HIP_R_8I, HIP_R_32I, HIP_R_32I,
            HIPBLAS_COMPUTE_32I, HIP_R_32I, 1, 1, 4, 4};
  if (name == "fp8-fp8")
    return {HIP_R_8F_E4M3, HIP_R_8F_E4M3, HIP_R_16F, HIP_R_16F,
            HIPBLAS_COMPUTE_32F, HIP_R_32F, 1, 1, 2, 2};
  if (name == "fp8-bf8")
    return {HIP_R_8F_E4M3, HIP_R_8F_E5M2, HIP_R_16F, HIP_R_16F,
            HIPBLAS_COMPUTE_32F, HIP_R_32F, 1, 1, 2, 2};
  fail("unsupported type " + std::string(name));
}
struct Buffer {
  void *p = nullptr;
  explicit Buffer(size_t bytes) { hip_check(hipMalloc(&p, bytes), "hipMalloc"); }
  ~Buffer() { if (p) (void)hipFree(p); }
};
struct Event {
  hipEvent_t e = nullptr;
  Event() { hip_check(hipEventCreate(&e), "hipEventCreate"); }
  ~Event() { if (e) (void)hipEventDestroy(e); }
};
float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  size_t i = values.size() / 2;
  return values.size() & 1 ? values[i] : (values[i - 1] + values[i]) / 2;
}

struct Descriptors {
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  hipblasLtMatmulDesc_t op = nullptr;
  Descriptors(const Options &o, const Type &t) {
    lt_check(hipblasLtMatrixLayoutCreate(&a, t.a, o.m, o.k, o.m), "A layout");
    lt_check(hipblasLtMatrixLayoutCreate(&b, t.b, o.k, o.n, o.k), "B layout");
    lt_check(hipblasLtMatrixLayoutCreate(&c, t.c, o.m, o.n, o.m), "C layout");
    lt_check(hipblasLtMatrixLayoutCreate(&d, t.d, o.m, o.n, o.m), "D layout");
    lt_check(hipblasLtMatmulDescCreate(&op, t.compute, t.scale), "operation");
    hipblasOperation_t trans = HIPBLAS_OP_N;
    lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &trans, sizeof(trans)), "transA");
    lt_check(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &trans, sizeof(trans)), "transB");
  }
  ~Descriptors() {
    if (op) hipblasLtMatmulDescDestroy(op);
    if (a) hipblasLtMatrixLayoutDestroy(a);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (c) hipblasLtMatrixLayoutDestroy(c);
    if (d) hipblasLtMatrixLayoutDestroy(d);
  }
};

} // namespace

int main(int argc, char **argv) try {
  Options o = options(argc, argv);
  Type t = type_for(o.type);
  hip_check(hipSetDevice(o.device), "hipSetDevice");
  hipDeviceProp_t prop{};
  hip_check(hipGetDeviceProperties(&prop, o.device), "device properties");
  hipStream_t stream = nullptr;
  hip_check(hipStreamCreate(&stream), "stream create");
  hipblasLtHandle_t handle = nullptr;
  lt_check(hipblasLtCreate(&handle), "handle create");
  Descriptors desc(o, t);
  Buffer a(static_cast<size_t>(o.m * o.k) * t.a_size);
  Buffer b(static_cast<size_t>(o.k * o.n) * t.b_size);
  Buffer c(static_cast<size_t>(o.m * o.n) * t.c_size);
  Buffer d(static_cast<size_t>(o.m * o.n) * t.d_size);
  Buffer workspace(o.workspace);
  hip_check(hipMemset(a.p, 1, static_cast<size_t>(o.m * o.k) * t.a_size), "init A");
  hip_check(hipMemset(b.p, 1, static_cast<size_t>(o.k * o.n) * t.b_size), "init B");
  hip_check(hipMemset(c.p, 0, static_cast<size_t>(o.m * o.n) * t.c_size), "init C");
  float alpha_f = 1, beta_f = 0;
  int32_t alpha_i = 1, beta_i = 0;
  const void *alpha = o.type == "i8" ? static_cast<void *>(&alpha_i) : static_cast<void *>(&alpha_f);
  const void *beta = o.type == "i8" ? static_cast<void *>(&beta_i) : static_cast<void *>(&beta_f);
  std::vector<hipblasLtMatmulHeuristicResult_t> algorithms;
  lt_check(hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                      HIPBLAS_OP_N, HIPBLAS_OP_N, t.a, t.b, t.c,
                                      t.d, t.compute, algorithms), "getAllAlgos");
  std::cout << "{\n  \"schema\":\"loom-blas.route-probe.v1\",\n  \"device\":{\"ordinal\":"
            << o.device << ",\"name\":\"" << escape(prop.name) << "\",\"arch\":\""
            << escape(prop.gcnArchName) << "\"},\n  \"request\":{\"type\":\"" << o.type
            << "\",\"m\":" << o.m << ",\"n\":" << o.n << ",\"k\":" << o.k
            << "},\n  \"enumerated_count\":" << algorithms.size() << ",\n  \"eligible\":[\n";
  bool first = true;
  size_t eligible = 0;
  for (auto &candidate : algorithms) {
    size_t needed = 0;
    hipblasStatus_t support = hipblaslt_ext::matmulIsAlgoSupported(
        handle, desc.op, alpha, desc.a, desc.b, beta, desc.c, desc.d,
        candidate.algo, needed);
    if (support != HIPBLAS_STATUS_SUCCESS || needed > o.workspace) continue;
    ++eligible;
    auto algorithm = candidate.algo;
    auto launch = [&] {
      lt_check(hipblasLtMatmul(handle, desc.op, alpha, a.p, desc.a, b.p, desc.b,
                              beta, c.p, desc.c, d.p, desc.d, &algorithm,
                              workspace.p, needed, stream), "matmul");
    };
    for (int i = 0; i < o.warmup; ++i) launch();
    hip_check(hipStreamSynchronize(stream), "warmup");
    Event start, stop;
    std::vector<float> times;
    for (int i = 0; i < o.iterations; ++i) {
      hip_check(hipEventRecord(start.e, stream), "start"); launch();
      hip_check(hipEventRecord(stop.e, stream), "stop");
      hip_check(hipEventSynchronize(stop.e), "sync");
      float ms = 0; hip_check(hipEventElapsedTime(&ms, start.e, stop.e), "elapsed");
      times.push_back(ms * 1000);
    }
    if (!first) std::cout << ",\n";
    first = false;
    std::cout << "    {\"index\":" << hipblaslt_ext::getIndexFromAlgo(algorithm)
              << ",\"solution_name\":\"" << escape(hipblaslt_ext::getSolutionNameFromAlgo(handle, algorithm))
              << "\",\"kernel_name\":\"" << escape(hipblaslt_ext::getKernelNameFromAlgo(handle, algorithm))
              << "\",\"workspace_bytes\":" << needed << ",\"median_us\":"
              << std::fixed << std::setprecision(3) << median(times) << "}";
  }
  std::cout << "\n  ],\n  \"eligible_count\":" << eligible << "\n}\n";
  lt_check(hipblasLtDestroy(handle), "handle destroy");
  hip_check(hipStreamDestroy(stream), "stream destroy");
  return 0;
} catch (const std::exception &e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
