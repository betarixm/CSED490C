// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <gputk.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define gpuTKCheck(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                           \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));        \
      return -1;                                                               \
    }                                                                          \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  __shared__ float section[BLOCK_SIZE * 2];

  int inputIdx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

  if (inputIdx < len) {
    section[threadIdx.x] = input[inputIdx];
  }

  if (blockDim.x + inputIdx < len) {
    section[blockDim.x + threadIdx.x] = input[blockDim.x + inputIdx];
  }

  for (unsigned stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();

    int index = (threadIdx.x + 1) * stride * 2 - 1;

    if (index < BLOCK_SIZE * 2) {
      section[index] += section[index - stride];
    }
  }

  for (unsigned stride = BLOCK_SIZE * 2 / 4; stride > 0; stride /= 2) {
    __syncthreads();

    int index = (threadIdx.x + 1) * stride * 2 - 1;

    if (index + stride < BLOCK_SIZE * 2) {
      section[index + stride] += section[index];
    }
  }

  __syncthreads();

  if (inputIdx < len) {
    output[inputIdx] = section[threadIdx.x];
  }

  if (blockDim.x + inputIdx < len) {
    output[blockDim.x + inputIdx] = section[blockDim.x + threadIdx.x];
  }
}

__global__ void merge(float *data, int len) {
  int sectionSize = blockDim.x;

  for (int sectionIdx = 1; sectionIdx < ceil(float(len) / sectionSize);
       sectionIdx++) {
    __syncthreads();

    int dataIdx = sectionIdx * sectionSize + threadIdx.x;
    int base = sectionIdx * sectionSize - 1;

    if (0 <= base && dataIdx < len) {
      data[dataIdx] = data[dataIdx] + data[base];
    }
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The number of input elements in the input is ", numElements);

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  gpuTKCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  gpuTKCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Clearing output memory.");
  gpuTKCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  gpuTKTime_stop(GPU, "Clearing output memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  gpuTKCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                        cudaMemcpyHostToDevice));
  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(ceil((float)numElements / blockDim.x));

  gpuTKTime_start(Compute, "Performing CUDA computation");
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, numElements);
  merge<<<1, BLOCK_SIZE * 2>>>(deviceOutput, numElements);

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  gpuTKCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
