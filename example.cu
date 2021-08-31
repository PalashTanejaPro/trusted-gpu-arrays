#include "kmeans.h"
#include <vector>
#include <cstdio>
#include "./SecureCudaArray.hu"
#include <cmath>
#include <iostream>


/* 
Kmeans algorithm
Assignment step: 
    Assign each observation to the cluster with the nearest mean: that with the least squared Euclidean distance.
    
    This is done by get_nearest_centroid functions

Update step:
     Recalculate means (centroids) for observations assigned to each cluster.

    This is done by get_new_centroid functions

Also created "_shared" versions of these functions to speed up computations by using the architectural caches on GPUs

Unfortunately the version of Cuda my university uses only allows for C-style coding, therefore some of this code is less modularized than I would like it to be.
*/
__device__ float distance(float* __restrict__ a, float* __restrict__ b, int dim) {
    float sum = 0;
    for(int i = 0; i < dim; i++)
    {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    
    return sqrt(sum);
}

__global__ void get_nearest_centroid(float* __restrict__ data,
                                     float* __restrict__ centroids, 
                                     float* __restrict__ centroid_sum,
                                     int* __restrict__ centroid_count,
                                     int* __restrict__ cluster_assignment,
                                     int N,
                                     int k,
                                     int dim) {
    // similar to prefix sum, every thread gets all the data and works on their specific portion of it
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float* p = data + (idx * dim);

        float min_centroid_dist = distance(centroids, p, dim);
        int chosen = 0;

        for(int i = 1; i < k; ++i) {
            float dist = distance(centroids + (dim * i), p, dim);
            if (dist < min_centroid_dist) {
                chosen = i;
                min_centroid_dist = dist;
            }
        }

        for(int i = 0; i < dim; ++i) {
            atomicAdd((centroid_sum + (chosen * dim) + i), p[i]);
        }

        atomicAdd(&centroid_count[chosen], 1);
        cluster_assignment[idx] = chosen;
    }
}

__global__ void get_nearest_centroid_shared(float* __restrict__ data,
                                     float* __restrict__ centroids_global, 
                                     float* __restrict__ centroid_sum,
                                     int* __restrict__ centroid_count,
                                     int* __restrict__ cluster_assignment,
                                     int N,
                                     int k,
                                     int dim) {
    extern __shared__ float centroids[];

    // similar to prefix sum, every thread gets all the data and works on their specific portion of it
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= N) return;

    // use first few threads to copy over centroids
    if (threadIdx.x < k) {
        for(int i = 0; i < dim; ++i) {
            centroids[(threadIdx.x * dim) + i] = centroids_global[(threadIdx.x * dim) + i];
        }
    }
    __syncthreads();

    // doesn't make sense to copy over point to shared memory because it's specific to thread
    // instead copy it over to local memory to prevent global memory accesses
    float p[1024];
    for(int i = 0; i < dim; ++i) {
        p[i] = *(data + (idx * dim) + i);
    }

    float min_centroid_dist = distance(centroids, p, dim);
    int chosen = 0;

    for(int i = 1; i < k; ++i) {
        float dist = distance(centroids + (dim * i), p, dim);
        if (dist < min_centroid_dist) {
            chosen = i;
            min_centroid_dist = dist;
        }
    }

    for(int i = 0; i < dim; ++i) {
        atomicAdd((centroid_sum + (chosen * dim) + i), p[i]);
    }

    atomicAdd(&centroid_count[chosen], 1);
    cluster_assignment[idx] = chosen;

}


__global__ void get_new_centroids_shared(float* __restrict__ centroids_global,
                                  float* __restrict__ centroid_sum_global,
                                  int* __restrict__ centroid_count_global,
                                  bool* __restrict__ repeat,
                                  float threshold,                                
                                  int k,
                                  int dim) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > k) return;

    // 3D array with array 0 being centroids, array 1 being centroid sums, array 2 being centroid counts
    extern __shared__ float centroids[];

    if (threadIdx.x < k) {
        for(int i = 0; i < dim; ++i) {
            centroids[threadIdx.x * dim + i] = centroids_global[threadIdx.x * dim + i];
            centroids[(k * dim) + threadIdx.x * dim + i] = centroid_sum_global[threadIdx.x * dim + i];
        }
        centroids[(2 * k * dim) + threadIdx.x] = centroid_count_global[threadIdx.x];
    }
    __syncthreads();

    for(int i = 0; i < dim; ++i) {
        // easy shared mem optimization
        float cur_centroid = centroids[idx * dim + i];
        centroids[idx * dim + i]  = centroids[(k * dim) + idx * dim + i] / (float) centroids[(2 * k * dim) + idx];     
        if(abs(cur_centroid - centroids[idx * dim + i]) > threshold) {
            *repeat = true;
        }
    }
    
    __syncthreads();

    if (threadIdx.x < k) {
        for(int i = 0; i < dim; ++i) {
            centroids_global[threadIdx.x * dim + i] = centroids[threadIdx.x * dim + i];
            centroid_sum_global[threadIdx.x * dim + i] = centroids[(k * dim) + threadIdx.x * dim + i];
        }
        centroid_count_global[threadIdx.x] = centroids[(2 * k * dim) + threadIdx.x];
    }
    __syncthreads();
}

__global__ void get_new_centroids(float* __restrict__ centroids,
                                  float* __restrict__ centroid_sum,
                                  int* __restrict__ centroid_count,
                                  bool* __restrict__ repeat,
                                  float threshold,
                                  int k,
                                  int dim) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < k) {
        for(int i = 0; i < dim; ++i) {
            float cur_centroid = *(centroids + (idx * dim) + i);
            *(centroids + (idx * dim) + i)  = *(centroid_sum + (idx * dim) + i) / (float) centroid_count[idx];     
            if (abs(cur_centroid - *(centroids + (idx * dim) + i)) > threshold) {
                *repeat = true;
            }  
        }
        
    }
}

__global__ void find_nearest_centroid(float* __restrict__ data,
                                      float* __restrict__ centroids_global,
                                      float* __restrict__ min_dist,
                                      int N, 
                                      int k,
                                      int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    extern __shared__ float centroids[];
    if (threadIdx.x < k) {
        for(int i = 0; i < dim; ++i) {
            centroids[(threadIdx.x * dim) + i] = centroids_global[(threadIdx.x * dim) + i];
        }
    }
    __syncthreads();

    float* p = data + (idx * dim);
    float min_centroid_dist = distance(centroids, p, dim);

    for(int i = 1; i < k; ++i) {
        min_centroid_dist = min(distance(centroids + (dim * i), p, dim), min_centroid_dist);
    }

    min_dist[idx] = min_centroid_dist;
}

vector<float> kmeans_plus_plus(vector<Point> points, Point* centroids, int k) {
    int N = points.size();
    int dim = points[0].coord.size();

    float flat_points[N * dim];
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < dim; ++j) {
            flat_points[i * dim + j] = points[i].coord[j];
        }
    }
    SecureCudaArray<float>* cuda_data = new SecureCudaArray<float>(N * dim);
    cuda_data->set(flat_points, N * dim);

    float flat_centroids[k * dim];
    for(int i = 0; i < k; ++i) {
        for(int j = 0; j < dim; ++j) {
            flat_centroids[i * dim + j] = centroids[i].coord[j];
        }
    }
    SecureCudaArray<float>* cuda_centroids = new SecureCudaArray<float>(k * dim);
    cuda_centroids->set(flat_centroids, k * dim);

    SecureCudaArray<float>* cuda_min_dist = new SecureCudaArray<float>(N);

    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;

    find_nearest_centroid<<<threads, blocks, k * dim * sizeof(float)>>>(cuda_data->getData(),
                                                                         cuda_centroids->getData(),
                                                                         cuda_min_dist->getData(),
                                                                         N,
                                                                         k,
                                                                         dim);
    
    float min_dist_device[N];
    cuda_min_dist->get(min_dist_device, N);

    delete cuda_min_dist;
    delete cuda_centroids;
    delete cuda_data;

    return vector<float>(min_dist_device, min_dist_device + N);
}

KMeans kmeans_cuda(vector<Point> points, Point* centroids, int k, int max_iterations, float threshold, bool shared) {
    // **** Memory ops

    int N = points.size();
    int dim = points[0].coord.size();

    float* flat_points = (float*) malloc(sizeof(float) * N * dim);
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < dim; ++j) {
            flat_points[i * dim + j] = points[i].coord[j];
        }
    }
    SecureCudaArray<float>* cuda_data = new SecureCudaArray<float>(N * dim);
    cuda_data->set(flat_points, N * dim);
    free(flat_points);

    float flat_centroids[k * dim];
    for(int i = 0; i < k; ++i) {
        for(int j = 0; j < dim; ++j) {
            flat_centroids[i * dim + j] = centroids[i].coord[j];
        }
    }
    SecureCudaArray<float>* cuda_centroids = new SecureCudaArray<float>(k * dim);
    cuda_centroids->set(flat_centroids, k * dim);

    SecureCudaArray<int>* centroid_counts = new SecureCudaArray<int>(k);
    SecureCudaArray<int>* cluster_assignment = new SecureCudaArray<int>(N);
    SecureCudaArray<float>* centroid_sum = new SecureCudaArray<float>(k * dim);
    cluster_assignment->fillZeroes();

    // End of memory ops

    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // *** Main loop
    int iterations = 0;
    float total_time = 0;
    for(int i = 0; i < max_iterations; ++i) {
        centroid_counts->fillZeroes();
        centroid_sum->fillZeroes();

        cudaEventRecord(start);
        if (shared) {
            get_nearest_centroid_shared<<<blocks, threads, k * dim * sizeof(float)>>>(cuda_data->getData(),
                                        cuda_centroids->getData(),
                                        centroid_sum->getData(),
                                        centroid_counts->getData(),
                                        cluster_assignment->getData(),
                                        N,
                                        k,
                                        dim);
        } else {
            get_nearest_centroid<<<blocks, threads>>>(cuda_data->getData(),
                                        cuda_centroids->getData(),
                                        centroid_sum->getData(),
                                        centroid_counts->getData(),
                                        cluster_assignment->getData(),
                                        N,
                                        k,
                                        dim);
        }
        
        cudaDeviceSynchronize();

        bool* repeat;
        cudaMalloc(&repeat, sizeof(bool));
        if (shared) {
            get_new_centroids_shared<<<1, k, ((k * dim * 2) + k) * sizeof(float)>>>(cuda_centroids->getData(),
                                    centroid_sum->getData(),
                                    centroid_counts->getData(),
                                    repeat,
                                    threshold,
                                    k,
                                    dim);
        } else {
            get_new_centroids<<<1, k>>>(cuda_centroids->getData(),
                                    centroid_sum->getData(),
                                    centroid_counts->getData(),
                                    repeat,
                                    threshold,
                                    k,
                                    dim);
        }
        

        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;

        iterations++;

        bool repeat_loop;
        cudaMemcpy(&repeat_loop, repeat, sizeof(bool), cudaMemcpyDeviceToHost);
        if (!repeat_loop) {
            break;
        }
    }

    cuda_centroids->get(flat_centroids, k * dim);
    for(int i = 0; i < k; ++i) {
        for(int j = 0; j < dim; ++j) {
            centroids[i].coord[j] = flat_centroids[i * dim + j];
        }
    }

    int assignments[N];
    cluster_assignment->get(assignments, N);

    delete cluster_assignment;
    delete cuda_centroids;
    delete cuda_data;
    delete centroid_counts;
    delete centroid_sum;

    return KMeans{vector<Point>(centroids, centroids + k), vector<int>(assignments, assignments + N), total_time, iterations};
}