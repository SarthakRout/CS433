#include<iostream>
#include<omp.h>
#include<cassert>
#include<chrono>
#include<iomanip>

using namespace std;

void init(float* mat, float* x, float*y, long long n, long long t){
    #pragma omp parallel for num_threads(t)   
    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            mat[i*n + j] = (1.0 + i + j)/(n + 4);
        }
        x[i] = (1.0*i)/(i + 7);
        y[i] = 0.0;
    }
}

void update(float* mat, float* x, float* y, long long n, long long t){
    #pragma omp parallel for num_threads(t)
    for(int i = 0; i<n; i++){
        float party = 0.0;
        for(int j = 0; j<n; j++){
            party += mat[n*i + j]*x[j];
        }
        y[i] = party; 
    }
}

float matmul(float * mat, float* x, float* y, long long n){
    unsigned long long i, j;
    float diff = 0.0;
    for(i =0; i<n; i++){
        float y_ = 0.0;
        for(j = 0; j<n; j++){
            y_ += mat[i*(n) + j]*x[j]; 
        }
        diff += abs(y[i] - y_);
    }
}

int main(int argc, char** argv){
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
    float *mat, *x, *y;
    mat = (float *)malloc((n)*(n)*sizeof(float));
    x =  (float *)malloc((n)*sizeof(float));
    y =  (float *)malloc((n)*sizeof(float));
    init(mat, x, y, n ,t);
    auto solve_start = std::chrono::high_resolution_clock::now();
    update(mat, x, y, n, t);
    auto solve_end = std::chrono::high_resolution_clock::now();
    float diff = matmul(mat, x, y, n);
    double solve_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); 
    solve_time_taken *= 1e-9;
    cout<<"Time Taken: "<<solve_time_taken<<"\nAverage Absolute Diff: "<<(diff/n)<<"\n";
    free(mat);
    free(x);
    free(y);
}