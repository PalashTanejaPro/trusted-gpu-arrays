#ifndef _CUDA_ARRAY_H
#define _CUDA_ARRAY_H

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include "AES/AESCPU.h"
#include "AES.h"

#define AES_BLOCK_SIZE 16
#define AES_BITS 128

template <class T>
class SecureCudaArray
{
public:
    explicit SecureCudaArray()
        : start_(0),
          end_(0),
          key { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f }
    {
        aesCPU = new AES(AES_BITS);
        encryptedSize = 0;
    }

    explicit SecureCudaArray(size_t size)
    {
        size_t padding = size % AES_BLOCK_SIZE;
        size += padding;
        allocate(size);

        aesCPU = new AES(AES_BITS);
        encryptedSize = size;
    }

    ~SecureCudaArray()
    {
        free();
    }

    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    size_t getSize() const
    {
        return end_ - start_;
    }

    const T* getData() const
    {
        if (encryptedSize == 0) return start_;

        decryptdemo(key, (uint8_t*) start_, encryptedSize);
        encryptedSize = 0;
        return start_;
    }

    T* getData()
    {
        if (encryptedSize == 0) return start_;

        decryptdemo(key, (uint8_t*) start_, encryptedSize * sizeof(T));
        encryptedSize = 0;
        return start_;
    }

    void set(const T* src, size_t size)
    {
        unsigned int outLen;
        auto out = aesCPU->EncryptECB((unsigned char*) src, size * sizeof(T), key, outLen);
        cudaError_t result = cudaMemcpy(start_, out, size * sizeof(T), cudaMemcpyHostToDevice);

        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }
    void get(T* dest, size_t size)
    {
        cudaError_t result = cudaMemcpy(dest, start_, this->encryptedSize, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }

    void fillZeroes() {
        cudaMemset(start_, 0, getSize() * sizeof(T));
    }

private:
    void allocate(size_t size)
    {
        cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
        if (result != cudaSuccess)
        {
            start_ = end_ = 0;
            throw std::runtime_error("failed to allocate device memory");
        }
        end_ = start_ + size;
    }

    void free()
    {
        if (start_ != 0)
        {
            cudaFree(start_);
            start_ = end_ = 0;
        }
    }

    T* start_;
    T* end_;
    // in a real system, there would be a mechanism for the GPU and CPU to decide on a key at start
    unsigned char key[16];
    AES* aesCPU;
    uint32_t encryptedSize;
};

#endif