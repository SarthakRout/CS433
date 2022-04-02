#include<iostream>
#include<fstream>
#include<chrono>
#include<iomanip>

using namespace std;

int main() {
    
    int num_threads = 8;
    int N = 1e5;

    //Start measuring time
    auto solve_start = std::chrono::high_resolution_clock::now();

    int i = 0;
    #pragma omp parallel num_threads (num_threads) private (i)
    {
        for (i=0; i<N; i++) {
        // OMP Barrier
        #pragma omp barrier
        }
    }
    // Stop measuring time
    auto solve_end =  std::chrono::high_resolution_clock::now();

    //Report measuring time
    double solve_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); 
    solve_time_taken *= 1e-9;
    cout<<"Threads: "<<num_threads<<" Time: "<<solve_time_taken<<"\n";
    
    // Keeping Records
    #ifdef SARTHAK
    fstream rec;
    rec.open("records_barrier.txt", ios::app);
    rec<<"5"<<","<<num_threads<<","<<solve_time_taken<<"\n";
    rec.close();
    #endif
    return 0;
}