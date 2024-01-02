#include <gputk.h>

#define gpuTKCheck(stmt)                                                  \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                      \
      gpuTKLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));   \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Insert code to implement SPMV using JDS with transposed input here

// #define USING_SHARED_MEMORY 1

__global__ void spmvWithTransposedJds(float *data, int *columnIndices,
                                      int *columnPointers,
                                      int *rowPermutation, float *x,
                                      float *y, int numberOfRows,
                                      int lenColumnPointers) {

#ifdef USING_SHARED_MEMORY
  extern __shared__ char sharedMemory[];

  int *loadedColumnPointers = (int *)sharedMemory;

  for (int i = threadIdx.x; i < lenColumnPointers; i += blockDim.x) {
    loadedColumnPointers[i] = columnPointers[i];
  }

  __syncthreads();
#endif
#ifndef USING_SHARED_MEMORY
  int *loadedColumnPointers = columnPointers;
#endif

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < numberOfRows) {
    float dot = 0;

    for (int col = 0;
         row < loadedColumnPointers[col + 1] - loadedColumnPointers[col];
         col++) {
      dot += data[loadedColumnPointers[col] + row] *
             x[columnIndices[loadedColumnPointers[col] + row]];
    }

    y[rowPermutation[row]] = dot;
  }
}

int main(int argc, char **argv) {
  gpuTKArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = gpuTKArg_read(argc, argv);

  gpuTKTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &numARows,
                               &numAColumns);
  hostB = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numBRows,
                               &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  gpuTKTime_stop(Generic, "Importing data and creating memory on host");

  gpuTKLog(TRACE, "The dimensions of A are ", numARows, " x ",
           numAColumns);
  gpuTKLog(TRACE, "The dimensions of B are ", numBRows, " x ",
           numBColumns);

  gpuTKTime_start(GPU, "Converting matrix A to JDS format (transposed).");
  //@@ Create JDS format data

  int nonZeroElements = 0;

  int *hostRowPermutation = (int *)malloc(numARows * sizeof(int));
  int *hostRowLengths     = (int *)malloc(numARows * sizeof(int));

  for (int i = 0; i < numARows; i++) {
    hostRowPermutation[i] = i;
  }

  for (int i = 0; i < numARows; i++) {
    hostRowLengths[i] = 0;
  }

  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numAColumns; j++) {
      if (hostA[i * numAColumns + j] != 0) {
        nonZeroElements++;
        hostRowLengths[i]++;
      }
    }
  }

  for (int i = 0; i < numARows; i++) {
    int maxIndex = i;

    for (int j = i + 1; j < numARows; j++) {
      if (hostRowLengths[j] > hostRowLengths[maxIndex]) {
        maxIndex = j;
      }
    }

    int temp                     = hostRowPermutation[i];
    hostRowPermutation[i]        = hostRowPermutation[maxIndex];
    hostRowPermutation[maxIndex] = temp;

    temp                     = hostRowLengths[i];
    hostRowLengths[i]        = hostRowLengths[maxIndex];
    hostRowLengths[maxIndex] = temp;
  }

  float *permutedA =
      (float *)malloc(numARows * numAColumns * sizeof(float));

  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numAColumns; j++) {
      permutedA[i * numAColumns + j] =
          hostA[hostRowPermutation[i] * numAColumns + j];
    }
  }

  float *hostData = (float *)malloc(nonZeroElements * sizeof(float));
  int *hostColumnIndices  = (int *)malloc(nonZeroElements * sizeof(int));
  int *hostColumnPointers = (int *)malloc((numAColumns + 1) * sizeof(int));

  hostColumnPointers[0] = 0;

  for (int i = 0; i < numARows; i++) {
    hostRowLengths[i] = 0;
  }

  int hostDataIndex = 0;

  int *columnIndexByRow = (int *)malloc(numARows * sizeof(int));

  for (int i = 0; i < numARows; i++) {
    columnIndexByRow[i] = 0;
  }

  for (int i = 0; i < numAColumns; i++) {
    int length = 0;
    for (int row = 0; row < numARows; row++) {
      for (int col = columnIndexByRow[row]; col < numAColumns; col++) {
        if (permutedA[row * numAColumns + col] != 0) {
          hostData[hostDataIndex] = permutedA[row * numAColumns + col];
          hostColumnIndices[hostDataIndex] = col;
          hostDataIndex++;
          columnIndexByRow[row] = col + 1;
          length++;
          break;
        }
      }
    }
    hostColumnPointers[i + 1] = hostColumnPointers[i] + length;
  }

  gpuTKTime_stop(GPU, "Converting matirx A to JDS format (transposed).");

  gpuTKTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int *deviceColumnIndices;
  int *deviceColumnPointers;
  int *deviceRowPermutation;

  cudaMalloc(&deviceA, nonZeroElements * sizeof(float));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));
  cudaMalloc(&deviceColumnIndices, nonZeroElements * sizeof(int));
  cudaMalloc(&deviceColumnPointers, (numAColumns + 1) * sizeof(int));
  cudaMalloc(&deviceRowPermutation, numARows * sizeof(int));

  gpuTKTime_stop(GPU, "Allocating GPU memory.");

  gpuTKTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostData, nonZeroElements * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceColumnIndices, hostColumnIndices,
             nonZeroElements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceColumnPointers, hostColumnPointers,
             (numAColumns + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceRowPermutation, hostRowPermutation,
             numARows * sizeof(int), cudaMemcpyHostToDevice);

  gpuTKTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(256);
  dim3 dimGrid(ceil((float)numARows / dimBlock.x));

  gpuTKTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  spmvWithTransposedJds<<<dimGrid, dimBlock,
                          (numAColumns + 1) * sizeof(int)>>>(
      deviceA, deviceColumnIndices, deviceColumnPointers,
      deviceRowPermutation, deviceB, deviceC, numARows, numAColumns + 1);

  cudaDeviceSynchronize();
  gpuTKTime_stop(Compute, "Performing CUDA computation");

  gpuTKTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
             cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying output memory to the CPU");

  gpuTKTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  cudaFree(deviceColumnIndices);
  cudaFree(deviceColumnPointers);
  cudaFree(deviceRowPermutation);

  gpuTKTime_stop(GPU, "Freeing GPU Memory");

  gpuTKSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
