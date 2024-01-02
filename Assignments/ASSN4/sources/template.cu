#include <gputk.h>

#define gpuTKCheck(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE

__global__ void convolute(float *imageInput, float *mask, float *imageOutput,
                          int imageChannels, int imageWidth, int imageHeight) {
  __shared__ float paddedImageTile[w][w];

  for (int channelIndex = 0; channelIndex < imageChannels; channelIndex++) {
    int paddedRow = threadIdx.y;
    int paddedCol = threadIdx.x;

    int inputTileRow = paddedRow - Mask_radius;
    int inputTileCol = paddedCol - Mask_radius;

    int inputRow = blockIdx.y * TILE_WIDTH + inputTileRow;
    int inputCol = blockIdx.x * TILE_WIDTH + inputTileCol;

    int inputIdx =
        (inputRow * imageWidth + inputCol) * imageChannels + channelIndex;

    if (0 <= inputRow && inputRow < imageHeight && 0 <= inputCol &&
        inputCol < imageWidth) {
      paddedImageTile[paddedRow][paddedCol] = imageInput[inputIdx];
    } else {
      paddedImageTile[paddedRow][paddedCol] = 0;
    }

    __syncthreads();

    int tileRow = threadIdx.y;
    int tileCol = threadIdx.x;

    int outputRow = blockIdx.y * TILE_WIDTH + tileRow;
    int outputCol = blockIdx.x * TILE_WIDTH + tileCol;
    int outputIdx =
        (outputRow * imageWidth + outputCol) * imageChannels + channelIndex;

    if (tileRow < TILE_WIDTH && tileCol < TILE_WIDTH) {
      float result = 0;

      for (int maskRow = 0; maskRow < Mask_width; maskRow++) {
        for (int maskCol = 0; maskCol < Mask_width; maskCol++) {
          result += paddedImageTile[tileRow + maskRow][tileCol + maskCol] *
                    mask[maskRow * Mask_width + maskCol];
        }
      }

      if (outputRow < imageHeight && outputCol < imageWidth) {
        imageOutput[outputIdx] = result;
      }
    }

    __syncthreads();
  }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile = gpuTKArg_getInputFile(arg, 1);

  inputImage = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth = gpuTKImage_getWidth(inputImage);
  imageHeight = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  cudaMalloc(&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc(&deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc(&deviceMaskData, maskRows * maskColumns * sizeof(float));

  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData,
             maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);

  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  dim3 blockDim(w, w);
  dim3 gridDim(ceil((float)imageWidth / TILE_WIDTH),
               ceil((float)imageHeight / TILE_WIDTH));

  convolute<<<gridDim, blockDim>>>(deviceInputImageData, deviceMaskData,
                                   deviceOutputImageData, imageChannels,
                                   imageWidth, imageHeight);

  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyDeviceToHost);
  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
