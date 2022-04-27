#include<iostream>
#include<cstdlib>
#include<chrono>
#include<cassert>
#include<iomanip>
#include<cuda.h>

#define THREADS_PER_BLOCK 64
#define X_SPLIT THREADS_PER_BLOCK

using namespace std;
float diff = 0;
__global__ void init(float* mat, float* x, float*y, long long n, int span){
    unsigned long long i;
    long long id = threadIdx.x + blockIdx.x*blockDim.x;
    for(i = span*id; i<span*(id+1); i++){
        mat[i] = (1.0 + i/n + i%(n))/(n + 4);
    }
    if(id < n){
        x[id] = (1.0*id)/(id + 7);
        y[id] = 0.0;
    }
}

__global__ void update2(float* mat, float*x,  float*y, long long n, long long t){
    long long id = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned long long i;
    long long span = (n*n)/t;
    long long cury = (span*id)/n;
    float party = 0.0; 
    for(i = span*id; i<span*(id+1); i++){
        
        party += mat[i]*x[i%n];
        if((i + 1)%n == 0){
            atomicAdd(&y[cury], party);
            cury += 1;
            party = 0.0;
        }
    }
    if(cury < n){
        atomicAdd(&y[cury], party);
    }
}


__global__ void update(float* mat, float*x,  float*y, long long n, long long t){
    long long id = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned long long idx;
    __shared__ float xv[X_SPLIT];
    for(int k = 0; k<n/t; k++){
        float party = 0.0;
        idx = k*t + id;
        for(int i = 0; i<n/X_SPLIT; i++){
            xv[threadIdx.x] = x[i*X_SPLIT + threadIdx.x];
            __syncthreads();
            for(int j = 0; j<X_SPLIT; j++){
                party += mat[idx*n + i*X_SPLIT + j]*xv[j];
            }
            __syncthreads();
        }
        y[idx] = party;
    }
}

void matmul(float * mat, float* x, float* y, long long n){
    unsigned long long i, j;
    for(i =0; i<n; i++){
        float y_ = 0.0;
        for(j = 0; j<n; j++){
            y_ += mat[i*(n) + j]*x[j]; 
        }
        diff += abs(y[i] - y_);
    }
}

int main(int argc, char**argv) {
    if(argc != 3){
        cout<<"Give exactly two arguments: the grid dimension (n) and the number of threads (t).\n";
        return -1;
    }
    long long n = atoll(argv[1]);
    long long t = atoll(argv[2]);
    if(t > n){
        cout<<"Currently, we support only t <= n .. taking t = n\n";
        t = n;
    }
    assert((n&(n-1)) == 0);
    assert((t&(t-1)) == 0);
    // Initialise the 2d array with random numbers
    float *mat, *x, *y;
    cudaMallocManaged((void**)&mat, (n)*(n)*sizeof(float));
    cudaMallocManaged((void**)&x, (n)*sizeof(float));
    cudaMallocManaged((void**)&y, (n)*sizeof(float));

    int device = -1;
    cudaGetDevice(&device);
	cudaMemAdvise(mat, sizeof(float)*(n)*(n), cudaMemAdviseSetAccessedBy, device);
	cudaMemAdvise(x, sizeof(float)*(n), cudaMemAdviseSetAccessedBy, device);
	cudaMemAdvise(y, sizeof(float)*(n), cudaMemAdviseSetAccessedBy, device);
    if(t < THREADS_PER_BLOCK){
        init<<<1, t>>>(mat, x, y, n, (n*n)/t);
    }
    else{
        init<<<t/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(mat, x, y, n, (n*n)/t);
    }
    // Run the solver kernel till convergence
    cudaDeviceSynchronize();
    auto solve_start = std::chrono::high_resolution_clock::now();
    if(t < THREADS_PER_BLOCK){
        update<<<1, t>>>(mat, x, y, n, t);
    }
    else{
        update<<<t/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(mat, x, y, n, t);
    }
    cudaDeviceSynchronize();
    auto solve_end = std::chrono::high_resolution_clock::now();
    matmul(mat, x, y, n);
    double solve_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); 
    solve_time_taken *= 1e-9;
    cout<<"Time Taken: "<<solve_time_taken<<"\nAverage Absolute Diff: "<<(diff/n)<<"\n";
    return 0;
}