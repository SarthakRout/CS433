/*
To Run:
g++  -std=c++1y -fopenmp ./omp_main_lock.cpp ./sync_library.hpp -o ./omp_main_lock; ./omp_main_lock
*/
#include<iostream>
#include<fstream>
#include<chrono>
#include<iomanip>
#include<cassert>

using namespace std;

int main() {
    
    int num_threads = 8;
    int N = 1e7;

    //Start measuring time
    auto solve_start = std::chrono::high_resolution_clock::now();

    int x = 0, y = 0, i = 0;
    #pragma omp parallel num_threads (num_threads) private (i)
    {
        for (i=0; i<N; i++) {
        // OMP Critical Section
        #pragma omp critical
            {
                assert (x == y);
                x = y + 1;
                y++;
            }
        }
    }
    // Stop measuring time
    auto solve_end =  std::chrono::high_resolution_clock::now();

    // Check whether values match
    assert(x == y);
    assert(x == N*num_threads);

    //Report measuring time
    double solve_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); 
    solve_time_taken *= 1e-9;
    cout<<"Threads: "<<num_threads<<" Time: "<<solve_time_taken<<"\n";
    
    // Keeping Records
    #ifdef SARTHAK
    fstream rec;
    rec.open("records_lock.txt", ios::app);
    rec<<"7"<<","<<num_threads<<","<<solve_time_taken<<"\n";
    rec.close();
    #endif
    return 0;
}