#include<iostream>
#include<chrono>
#include<fstream>
#include<iomanip>
#include<cassert>
#include<pthread.h>
#include "sync_library.hpp"

using namespace std; 

int x = 0;
int y = 0;
const int N = 1e6;
struct Lock* lock = NULL;
void* critical(void* arg){
    int tid = (int)(long long)arg;
    for(int i = 0; i<N; i++){
        Acquire(lock, tid);
        assert(x==y);
        x = y + 1;
        y++;
        Release(lock, tid);
    }
    return NULL;
}

int main(){
    // Choose number of threads. (Please keep it less than 128.)
    const int num_threads = 6;
    /* Use
    BAKERY_LOCK, SPIN_LOCK, TTS_LOCK, TICKET_LOCK, ARRAY_LOCK, MUTEX, SEMAP
    as the first argument to the Init function
    */
    lock = Init(SPIN_LOCK, num_threads);

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
    // Check whether values match
    assert(x == y);
    assert(x == N*num_threads);

    //Report measuring time
    double solve_time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); 
    solve_time_taken *= 1e-9;
    cout<<"Lock Type: "<<lock->lock_type[0]<<" Threads: "<<num_threads<<" Time: "<<solve_time_taken<<"\n";

    // Keeping Records
    #ifdef SARTHAK
    fstream rec;
    rec.open("records_lock.txt", ios::app);
    rec<<lock->lock_type[0]<<","<<num_threads<<","<<solve_time_taken<<"\n";
    rec.close();
    #endif
    
    // Free memory allocated to the lock pointer
    Free(lock);
    return 0;
}