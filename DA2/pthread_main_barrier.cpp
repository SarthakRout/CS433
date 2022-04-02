#include<iostream>
#include<chrono>
#include<fstream>
#include<iomanip>
#include<cassert>
#include<pthread.h>
#include "sync_library.hpp"

using namespace std; 

const int N = 1e4;
struct Barrier* barrier = NULL;
void* critical(void* arg){
    int tid = (int)(long long)arg;
    int local_sense = 0; // For CENTRAL_BUSY;
    for(int i = 0; i<N; i++){
        // cout<<tid<<" "<<i<<"\n"<<flush;
        Set_Barrier(barrier, &local_sense, tid);
    }
    return NULL;
}

int main(){
    // Choose number of threads. (Please keep it less than 128.)
    const int num_threads = 8;
    /* Use
    CENTRAL_BUSY, TREE_BUSY, CENTRAL_CONDV, TREE_CONDV, POSIX_BARRIER
    as the first argument to the Init_Barrier function
    */
    barrier = Init_Barrier(TREE_CONDV, num_threads);

    //Start measuring time
    auto solve_start = std::chrono::high_resolution_clock::now();

    // Create Threads
    pthread_t threads[num_threads];
    for(int i = 0; i<num_threads; i++){
        if(pthread_create(&threads[i], NULL, critical, (void*)((long long)i)) != 0){
            perror("Couldn't create thread");
            exit(1);
        }
    }

    // Join all threads
    for(int i = 0; i<num_threads; i++){
        pthread_join(threads[i], NULL);
    }

    // Stop measuring time
    auto solve_end =  std::chrono::high_resolution_clock::now();

    //Report measuring time
    double solve_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); 
    solve_time_taken *= 1e-9;
    cout<<"Barrier Type: "<<barrier->barrier_type[0]<<" Threads: "<<num_threads<<" Time: "<<solve_time_taken<<"\n";

    // Keeping Records
    #ifdef SARTHAK
    fstream rec;
    rec.open("records_barrier.txt", ios::app);
    rec<<barrier->barrier_type[0]<<","<<num_threads<<","<<solve_time_taken<<"\n";
    rec.close();
    #endif
    
    // Free memory allocated to the lock pointer
    Free_Barrier(barrier);
    return 0;
}