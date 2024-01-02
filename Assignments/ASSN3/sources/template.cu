#include <gputk.h>

#define NUM_STREAMS 16

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len) {
    out[index] = in1[index] + in2[index];
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;
  unsigned int numStreams;

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The input length is ", inputLength);

  gpuTKTime_start(GPU, "Allocating Pinned memory.");

  //@@ Allocate GPU memory here using pinned memory here
  cudaHostRegister(hostInput1, inputLength * sizeof(float),
                   cudaHostRegisterMapped);
  cudaHostRegister(hostInput2, inputLength * sizeof(float),
                   cudaHostRegisterMapped);
  cudaHostRegister(hostOutput, inputLength * sizeof(float),
                   cudaHostRegisterMapped);

  cudaMalloc(&deviceInput1, inputLength * sizeof(float));
  cudaMalloc(&deviceInput2, inputLength * sizeof(float));
  cudaMalloc(&deviceOutput, inputLength * sizeof(float));

  //@@ Create and setup streams
  numStreams = NUM_STREAMS;

  cudaStream_t *streams =
      (cudaStream_t *)malloc(numStreams * sizeof(cudaStream_t));

  for (unsigned int s = 0; s < numStreams; s++) {
    cudaStreamCreate(&(streams[s]));
  }

  //@@ Calculate data segment size of input data processed by each stream
  int segmentLength = ceil((float)inputLength / numStreams);

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Perform parallel vector addition with different streams.
  for (unsigned int s = 0; s < numStreams; s++) {
    //@@ Asynchronous copy data to the device memory in segments
    int streamSegmentLength = std::max(
        0, std::min(inputLength - (int)s * segmentLength, segmentLength));

    cudaMemcpyAsync(deviceInput1 + s * segmentLength,
                    hostInput1 + s * segmentLength,
                    streamSegmentLength * sizeof(float), cudaMemcpyHostToDevice,
                    streams[s]);

    cudaMemcpyAsync(deviceInput2 + s * segmentLength,
                    hostInput2 + s * segmentLength,
                    streamSegmentLength * sizeof(float), cudaMemcpyHostToDevice,
                    streams[s]);

    //@@ Calculate starting and ending indices for per-stream data

    //@@ Invoke CUDA Kernel
    //@@ Determine grid and thread block sizes (consider ococupancy)
    dim3 blockDim(256);
    dim3 gridDim(ceil((float)streamSegmentLength / blockDim.x));

    vecAdd<<<gridDim, blockDim, 0, streams[s]>>>(
        deviceInput1 + s * segmentLength, deviceInput2 + s * segmentLength,
        deviceOutput + s * segmentLength, segmentLength);

    //@@ Asynchronous copy data from the device memory in segments
    cudaMemcpyAsync(hostOutput + s * segmentLength,
                    deviceOutput + s * segmentLength,
                    streamSegmentLength * sizeof(float), cudaMemcpyDeviceToHost,
                    streams[s]);
  }

  //@@ Synchronize
  for (unsigned int s = 0; s < numStreams; s++) {
    cudaStreamSynchronize(streams[s]);
  }
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(GPU, "Freeing Pinned Memory");
  //@@ Destory cudaStream
  for (unsigned int s = 0; s < numStreams; s++) {
    cudaStreamDestroy(streams[s]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  cudaHostUnregister(hostInput1);
  cudaHostUnregister(hostInput2);
  cudaHostUnregister(hostOutput);

  gpuTKTime_stop(GPU, "Freeing Pinned Memory");

  gpuTKSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
