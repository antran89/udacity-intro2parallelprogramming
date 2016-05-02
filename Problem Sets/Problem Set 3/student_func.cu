/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "limits"

using namespace std;

/**
 * Reduce operation to find the minimum value of logLum.
 * This algorithm uses shared memory.
 * @params:
 *		d_in:	1D vector
 *		d_out:	reduced vector
 *		size: 	dimensions of d_in
 */
__global__ void shmem_min_reduce_kernel(const float* const d_in,
										float* d_out,
										int size)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockIdx.x * blockDim.x;	// we use only 1D kernel
	int tid = threadIdx.x;

	// load shared mem from global mem
	if (myId < size)
		sdata[tid] = d_in[myId];
	else
		sdata[tid] = 1e4;	// a max value

	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();	// to avoid data race
	}

	// only thread 0 in each block write data back into global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

/**
 * Reduce operation to find the maximum value of logLum.
 * This algorithm uses shared memory.
 * @params:
 *		d_in:	1D vector
 *		d_out:	reduced vector
 *		size: 	dimensions of d_in
 */
__global__ void shmem_max_reduce_kernel(const float* const d_in,
										float* d_out,
										int size)
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float sdata[];

	int myId = threadIdx.x + blockIdx.x * blockDim.x;	// we use only 1D kernel
	int tid = threadIdx.x;

	// load shared mem from global mem
	if (myId < size)
		sdata[tid] = d_in[myId];
	else
		sdata[tid] = -1e4;	// a min value

	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
		}
		__syncthreads();	// to avoid data race
	}

	// only thread 0 in each block write data back into global mem
	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0];
	}
}

/**
 * Find min and max logLum using reduce algorithm.
 * @params:
 * 			d_in: d_logLuminance
 * 			min_logLum
 * 			max_logLum
 * 			size: number of pixels
 */
void reduce_minmax(const float* d_in, float &min_logLum, float &max_logLum,
					int size)
{
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = ceil(size / (float)maxThreadsPerBlock);
	cout << "blocks = " << blocks << endl;
	cout << "threads = " << threads << endl;
	cout << "size = " << size << endl;

	// allocate GPU intermediate memory
	float *d_intermediate, *d_out;
	cudaMalloc(&d_intermediate, blocks * sizeof(float));
	cudaMalloc(&d_out, sizeof(float));

	// find min logLum
	// compute min in the first level of reduce
	shmem_min_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
			(d_in, d_intermediate, size);
	// now, we reduce it in the second level (only in 1 block).
	threads = blocks;
	blocks = 1;
	shmem_min_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
			(d_intermediate, d_out, threads);
	//min_logLum = *d_out;	// using cudaMemcpy
	cudaMemcpy(&min_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

	// find max logLum
	threads = maxThreadsPerBlock;
	blocks = ceil(size / (float)maxThreadsPerBlock);
	// compute min in the first level of reduce
	shmem_max_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
			(d_in, d_intermediate, size);
	// now, we reduce it in the second level (only in 1 block).
	threads = blocks;
	blocks = 1;
	shmem_max_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
			(d_intermediate, d_out, threads);
	//min_logLum = *d_out;	// using cudaMemcpy
	cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

	// free memory in this scope
	cudaFree(d_intermediate);
	cudaFree(d_out);
}

__global__ void simple_histogram(const float* d_in, unsigned int* d_out,
								const size_t numBins, const float min_logLum,
								const float range_logLum, const int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size)
		return;
	unsigned int bin = static_cast<unsigned int> (fminf(numBins-1,
			(d_in[tid] - min_logLum) / range_logLum * numBins));
	atomicAdd(&(d_out[bin]), 1);
}

/**
 * Hillis and Steele exclusive scan algorithm
 * step complexity: O(log(n))
 * work complexity: O(nlog(n))
 * This is step efficient (SE) algorithm.
 * @params:
 * 			g_odata: global output memory
 * 			g_idata: global input memory
 * 			n : size of 1D input
 */
__global__ void scan(unsigned int *g_odata, unsigned int* const g_idata, int n)
{
	extern __shared__ unsigned int temp[]; // allocated on invocation
	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	// load input into shared memory.
	// Exclusive scan: shift right by one and set first element to 0
	temp[thid] = (thid > 0) ? g_idata[thid - 1] : 0;
	__syncthreads();
	for( int offset = 1; offset < n; offset <<= 1 )
	{
		pout = 1 - pout; // swap double buffer indices
		pin = 1 - pout;
		if (thid >= offset)
			temp[pout*n + thid] += temp[pin*n + thid - offset];
		else
			temp[pout*n + thid] = temp[pin*n + thid];
		__syncthreads();
	}
	g_odata[thid] = temp[pout*n + thid]; // write output
}

/**
 * Prefix scan algorithm: Down-sweep algorithm.
 * This algorithm is work-efficient.
 * @params:
 * * 		g_odata: global output memory
 * 			g_idata: global input memory
 * 			n : size of 1D input
 */
__global__ void prescan(unsigned int *g_odata, unsigned int* const g_idata, int n)
{
	extern __shared__ unsigned int temp[];		// allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	temp[2*thid] = g_idata[2*thid]; 	// load input into shared memory
	temp[2*thid+1] = g_idata[2*thid+1];
	for (int d = n>>1; d > 0; d >>= 1) 	// build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) { temp[n - 1] = 0; } // clear the last element

	for (int d = 1; d < n; d *= 2) 		// traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	g_odata[2*thid] = temp[2*thid]; 	// write results to device memory
	g_odata[2*thid + 1] = temp[2*thid + 1];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	// Step 1
	// find min and max logLum
	int numPixels = numRows * numCols;
	reduce_minmax(d_logLuminance, min_logLum, max_logLum, numPixels);
	cout << "min logLum: " << min_logLum << endl;
	cout << "max logLum: " << max_logLum << endl;

	// Step 2
	// find the range
	float range_logLum = max_logLum - min_logLum;

	// Step 3
	// generate a histogram of all the values in the logLuminance channel
	unsigned int* d_bins;	// histogram
	cudaMalloc(&d_bins, numBins * sizeof(unsigned int));
	int threads = 1024;
	int blocks = ceil(numPixels / (float)threads);
	simple_histogram<<<blocks, threads>>>
		(d_logLuminance, d_bins, numBins, min_logLum, range_logLum, numPixels);

	// Step 4: exclusive scan to find cdf
	cout << "number of bins: " << numBins << endl;
//	blocks = 1;
//	threads = numBins;
//	scan<<<blocks, threads, 2 * threads * sizeof(int)>>>
//			(d_cdf, d_bins, numBins);

	blocks = 1;
	threads = numBins/2;
	prescan<<<blocks, threads, 2 * threads * sizeof(int)>>>
			(d_cdf, d_bins, numBins);

	cudaFree(d_bins);
}
