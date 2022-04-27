#include<iostream>
#include<cstdlib>
#include<chrono>
#include<cassert>
#include<iomanip>
#include<cuda.h>

#define THREADS_PER_BLOCK 16
#define TILE_SIZE 16
using namespace std;


const float TOL = 1e-5;
const int MAX_ITER = 1000;
__managed__  float diff = 0.0;
__managed__ float rands[16384];


__global__ void init(float* grid, long long n, long long span){
    long long id = threadIdx.x + blockIdx.x*blockDim.x;
    long long maxm = (n+2)*(n+2);
    for(long long i = span*id; i<min((id+1)*span, maxm); i++){
	    grid[i] = rands[i%(16384)]/2;
    }
} 
__global__ void upd2(float*grid, long long n, long long t){
    long long span = (n*n)/t;
    long long id = threadIdx.x + blockIdx.x * blockDim.x;
    long long x, y;
    float local_diff = 0.0, prev;
    for(long long i = span*id; i<min((id+1)*span, n*n); i++){
        x = i/n;
        y = i%n;
        prev = grid[(x+1)*(n+2) +y + 1];
        grid[(x+1)*(n+2) + y + 1] = 0.2*(grid[(x+1)*(n+2) + y + 1] + grid[(x+1)*(n+2) + y] + grid[(x+1)*(n+2) + y + 2] + grid[(x)*(n+2) + y + 1] + grid[(x+2)*(n+2) + y + 1]);
        local_diff += abs(grid[(x+1)*(n+2) +y + 1] - prev);
        // atomicAdd(&diff, abs(grid[(x+1)*(n+2) +y + 1] - prev));
    }
    // atomicAdd(&diff, local_diff);
    // // __syncthreads();
    unsigned mask = 0xffffffff;
    unsigned long long int i;
	for (i=warpSize/2; i>0; i=i/2){
        local_diff += __shfl_down_sync(mask, local_diff, i);
    }
	if ((threadIdx.x % warpSize) == 0){
        atomicAdd(&diff, local_diff);
    }
}

__global__ void upd(float* grid, long long n, long long t){
    long long base = (blockIdx.x*THREADS_PER_BLOCK + 1)*(n+2) + 1;
    __shared__ float mat[THREADS_PER_BLOCK+2][TILE_SIZE+2];
    float local_diff = 0.0, prev;
    for(int k = 0; k<n/t; k++){
        for(int i = 0; i<n/TILE_SIZE; i++){
            for(int j = 0; j<TILE_SIZE+2; j++){
                mat[threadIdx.x+1][j] = grid[base + i*TILE_SIZE + threadIdx.x*(n+2) + j - 1];
            }
            for(int j = threadIdx.x; j<TILE_SIZE; j+=THREADS_PER_BLOCK){
                mat[0][j+1] = grid[base + i*TILE_SIZE - (n+2) + j];
                mat[THREADS_PER_BLOCK+1][j+1] = grid[base + i*TILE_SIZE + THREADS_PER_BLOCK*(n+2) + j] ;
            }
            __syncthreads();
            for(int j = 1; j<=TILE_SIZE; j++){
                prev = mat[threadIdx.x+1][j];
                mat[threadIdx.x+1][j] = 0.2*(mat[threadIdx.x+1][j] + mat[threadIdx.x+1][j-1] + mat[threadIdx.x+1][j+1] + mat[threadIdx.x][j]+ mat[threadIdx.x+2][j]);
                grid[base + i*TILE_SIZE + threadIdx.x*(n+2) + j - 1] = mat[threadIdx.x+1][j]; 
                local_diff += abs(mat[threadIdx.x+1][j] - prev);
            }
            __syncthreads();
        }
        base += (n+2)*t;
        // atomicAdd(&diff, local_diff);
    }
    
    unsigned mask = 0xffffffff;
    unsigned long long int i;
	for (i=warpSize/2; i>0; i=i/2){
        local_diff += __shfl_down_sync(mask, local_diff, i);
    }
	if ((threadIdx.x % warpSize) == 0){
        atomicAdd(&diff, local_diff);
    }
}

int main(int argc, char**argv) {
    if(argc != 3){
        std::cout<<"Give exactly two arguments: the grid dimension (n) and the number of threads (t).\n";
        return -1;
    }
    long long n = atoll(argv[1]);
    long long t = atoll(argv[2]);
    if(t > n){
        cout<<"Currently, we support only t <= n .. taking t = n\n";
        t = n;
    }
    long long init_span = ((n+2)*(n+2))/t;;
    if ((n+2)*(n+2) % t != 0){
        init_span++;
    }
    assert((n&(n-1)) == 0);
    assert((t&(t-1)) == 0);
    for(int i = 0; i<16384; i++){
    	rands[i] = (float)(random() %100)/100.0;
    }
    // Initialise the 2d array with random numbers
    float *grid;
    cudaMallocManaged((void**)&grid, (n+2)*(n+2)*sizeof(float));
    int device = -1;
    cudaGetDevice(&device);
	cudaMemAdvise(grid, sizeof(float)*(n+2)*(n+2), cudaMemAdviseSetAccessedBy, device);
    if(t < THREADS_PER_BLOCK){
        init<<<1, t>>>((float*)grid, n, init_span);
    }
    else{
        init<<<t/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>((float*)grid, n, init_span);
    }
    cudaDeviceSynchronize();
    // Run the solver kernel till convergence
    auto solve_start = std::chrono::high_resolution_clock::now();
    int itr;
    for(itr = 0; itr<MAX_ITER; itr++){
        if(t < THREADS_PER_BLOCK){
            upd<<<1, t>>>(grid, n, t);
        }
        else{
            upd<<<t/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(grid, n, t);
        }
        cudaDeviceSynchronize();
        if((diff/n)/n < TOL){
            break;
        }
        else{
            diff = 0.0;
        }
    }
    auto solve_end = std::chrono::high_resolution_clock::now();
    double solve_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); 
    solve_time_taken *= 1e-9;
    std::cout<<"Time Taken: "<<solve_time_taken<<"\nIterations: "<<(itr)<<"\nAverage Diff: "<<(diff/n)/n<<"\n";
    return 0;
}
