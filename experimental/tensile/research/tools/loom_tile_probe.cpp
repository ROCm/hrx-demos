#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void check_hip(hipError_t status, std::string_view operation) {
  if (status != hipSuccess) {
    fail(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

int parse_device(const char *text) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max()) {
    fail(std::string("invalid device: ") + text);
  }
  return static_cast<int>(value);
}

struct Options {
  int device = 0;
  std::filesystem::path hsaco;
  std::string kernel;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto next = [&]() -> const char * {
      if (++i >= argc) fail("missing value for " + std::string(arg));
      return argv[i];
    };
    if (arg == "--device") options.device = parse_device(next());
    else if (arg == "--hsaco") options.hsaco = next();
    else if (arg == "--kernel") options.kernel = next();
    else if (arg == "--help") {
      std::cout << "Usage: loom-tile-probe --device N --hsaco FILE "
                   "--kernel SYMBOL\n";
      std::exit(0);
    } else {
      fail("unknown option: " + std::string(arg));
    }
  }
  if (options.hsaco.empty()) fail("--hsaco is required");
  if (options.kernel.empty()) fail("--kernel is required");
  if (!std::filesystem::is_regular_file(options.hsaco) ||
      std::filesystem::file_size(options.hsaco) == 0) {
    fail("--hsaco must name a nonempty file");
  }
  return options;
}

struct DeviceBuffer {
  void *pointer = nullptr;
  explicit DeviceBuffer(size_t size) {
    check_hip(hipMalloc(&pointer, size), "hipMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() {
    if (pointer != nullptr) (void)hipFree(pointer);
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

void run(const Options &options) {
  check_hip(hipSetDevice(options.device), "hipSetDevice");
  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, options.device),
            "hipGetDeviceProperties");

  constexpr size_t kElementCount = 16U * 16U;
  std::vector<__half> host_input(kElementCount, __float2half(1.0F));
  std::vector<float> host_output(kElementCount, 0.0F);
  DeviceBuffer lhs(host_input.size() * sizeof(host_input[0]));
  DeviceBuffer rhs(host_input.size() * sizeof(host_input[0]));
  DeviceBuffer output(host_output.size() * sizeof(host_output[0]));
  check_hip(hipMemcpy(lhs.pointer, host_input.data(),
                      host_input.size() * sizeof(host_input[0]),
                      hipMemcpyHostToDevice),
            "copy lhs");
  check_hip(hipMemcpy(rhs.pointer, host_input.data(),
                      host_input.size() * sizeof(host_input[0]),
                      hipMemcpyHostToDevice),
            "copy rhs");
  check_hip(hipMemset(output.pointer, 0, host_output.size() * sizeof(float)),
            "clear output");

  Module module(options.hsaco);
  hipFunction_t function = nullptr;
  check_hip(hipModuleGetFunction(&function, module.value, options.kernel.c_str()),
            "hipModuleGetFunction");
  void *arguments[] = {&lhs.pointer, &rhs.pointer, &output.pointer};
  check_hip(hipModuleLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr,
                                  arguments, nullptr),
            "hipModuleLaunchKernel");
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  check_hip(hipMemcpy(host_output.data(), output.pointer,
                      host_output.size() * sizeof(float),
                      hipMemcpyDeviceToHost),
            "copy output");

  size_t mismatches = 0;
  double maximum_absolute = 0.0;
  for (const float value : host_output) {
    const double absolute = std::abs(static_cast<double>(value) - 16.0);
    maximum_absolute = std::max(maximum_absolute, absolute);
    if (absolute != 0.0) ++mismatches;
  }
  std::cout << "{\n"
            << "  \"schema\": \"loom-blas.loom-tile-probe.v1\",\n"
            << "  \"device\": {\"ordinal\": " << options.device
            << ", \"name\": \"" << properties.name << "\", \"arch\": \""
            << properties.gcnArchName << "\"},\n"
            << "  \"hsaco\": \"" << options.hsaco.string() << "\",\n"
            << "  \"kernel\": \"" << options.kernel << "\",\n"
            << "  \"mismatches\": " << mismatches << ",\n"
            << "  \"max_absolute\": " << std::setprecision(17)
            << maximum_absolute << "\n"
            << "}\n";
  if (mismatches != 0) fail("Loom tile output did not equal 16");
}

} // namespace

int main(int argc, char **argv) {
  try {
    run(parse_options(argc, argv));
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "loom-tile-probe: " << exception.what() << '\n';
    return 1;
  }
}
