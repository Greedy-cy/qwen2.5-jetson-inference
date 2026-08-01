#pragma once

#include "infer/common.hpp"

namespace infer {

class Buffer {     //RAII的资源封装  只关心起始地址，总字节数  在哪个设备
 public:
  Buffer() = default;
  Buffer(size_t bytes, Device device);
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  void reset();
  void resize(size_t bytes, Device device);
  void* data() { return data_; }
  const void* data() const { return data_; }
  size_t bytes() const { return bytes_; }
  Device device() const { return device_; }

 private:
  void* data_{nullptr};
  size_t bytes_{0};
  Device device_{Device::kCpu};
};

class Arena {           //buffer的线性分配器 维护一个游标记录当前分配到哪里了
 public:
  Arena() = default;
  Arena(size_t bytes, Device device) : storage_(bytes, device) {}
  void reset_cursor() { cursor_ = 0; }
  void* allocate(size_t bytes, size_t alignment = 256);
  size_t bytes() const { return storage_.bytes(); }
  size_t used() const { return cursor_; }
  Device device() const { return storage_.device(); }
 private:
  Buffer storage_;
  size_t cursor_{0};
};

}  // namespace infer
