<style type="text/css">
    ol { list-style-type: lower-alpha; }
</style>
In this assignment, you will study the performance of two important synchronization primitives used
in parallel programs namely, locks and barriers.

PART-A: LOCK DESIGN [60 POINTS]
-------------------------------

## Develop Acquire and Release functions for the following lock designs.

1. Lamport's Bakery lock (make sure to avoid false sharing)
2. Spin-lock employing cmpxchg instruction of x86
3. Test-and-test-and-set lock employing cmpxchg instruction of x86
4. Ticket lock
5. Array lock  

Notes: 

1. Assume cache block size of 64 bytes.
2. The spin-lock code is as discussed in the class (refer to slide no. 13)
3. Test-and-test-and-set lock code is as discussed in the class (refer to slide no. 16).
    You will have to develop a TestAndSet function using the cmpxchg instruction of x86. (Can also use xchg)
    This function will be similar to the CompareAndSet function developed in slide 13.
    You will use this TestAndSet function in place of the ts instruction.
    The test loop can be implemented in C/C++.
4. & e. You will need to develop a FetchAndInc function using the cmpxchg instruction.
          The rest of the code can be written in C/C++.
          For array lock, make sure to avoid false sharing.

You will compare the aforementioned five lock designs with POSIX mutex, a lock developed
using binary semaphores (refer to example discussed in class), and #pragma omp critical. For
conducting this study, you will use the following simple critical section involving two
shared variables x and y, both initialized to zero.

```cpp
{
   x = y + 1;
   y++;
}
```

Each thread executes this critical section ten million times (10^7). The code executed by
each thread is shown below where N=10^7.
```cpp
for (i=0; i<N; i++) {
   Acquire (&lock);
   assert (x == y);
   x = y + 1;
   y++;
   Release (&lock);
}
```

The overall code structure is shown below when using POSIX thread interface.

- Start timing measurement.
- Create threads.
- Each thread executes the aforementioned loop.
- Join threads.
- Stop timing measurement.
- assert (x == y);
- assert (x == N*t); // t is the number of threads
- Report measured time.

The overall code structure is shown below when measuring the efficiency of #pragma omp critical.

```cpp
Start timing measurement.
#pragma omp parallel num_threads (t) private (i)
{
   for (i=0; i<N; i++) {
#pragma omp critical
      {
         assert (x == y);
         x = y + 1;
         y++;
      }
   }
}
Stop timing measurement.
assert (x == y);
assert (x == N*t); // t is the number of threads
Report measured time.
```

Report the measured time of the eight locking techniques in a table as you vary the number
of threads from 1 to 64 in powers of two. Each row of the table should report the measured
times for a particular thread count (make eight columns in the table). Prepare a ninth column
of the table where you mention the best locking technique that you observe for each thread
count.

PART-B: BARRIER DESIGN [40 POINTS]
-----------------------------------

You will be comparing the performance of the following six barrier implementations.

1. Centralized sense-reversing barrier using busy-wait on flag (slide no. 52)
2. Tree barrier using busy-wait on flags (slide no. 56)
3. Centralized barrier using POSIX condition variable (slide no. 64)
4. Tree barrier using POSIX condition variable
5. POSIX barrier interface (pthread_barrier_wait)
6. #pragma omp barrier

Note: Use POSIX mutex locks in implementations (a), (c), (d).

You will conduct the performance measurement by executing a parallel program that has one million (10^6)
barriers and nothing else. The following loop is executed by each thread where N is one million.

```cpp
for (i=0; i<N; i++) {
   Barrier(...);
}
```

The overall code structure is shown below when using POSIX thread interface.

- Start timing measurement.
- Create threads.
- Each thread executes the aforementioned loop.
- Join threads.
- Stop timing measurement.
- Report measured time.

The overall code structure is shown below when measuring the efficiency of #pragma omp barrier.

```cpp
Start timing measurement.
#pragma omp parallel num_threads (t) private (i)
{
   for (i=0; i<N; i++) {
#pragma omp barrier
   }
}
Stop timing measurement.
Report measured time.
```

Report the measured time of the six barrier implementations in a table as you vary the number
of threads from 1 to 64 in powers of two. Each row of the table should report the measured
times for a particular thread count (make six columns in the table). Prepare a seventh column
of the table where you mention the best barrier implementation that you observe for each thread
count.

WHAT TO SUBMIT
---------------

Place all your Acquire, Release, and Barrier functions in a single C/C++ file named sync_library.x
where the .x extension should be chosen according to the language used. Name the functions differently
for different lock and barrier implementations. So, this file will have seven Acquire functions, seven
Release functions, and five Barrier functions. The omp critical and omp barrier do not require separate
functions. I have given a few examples below. Please put a comment specifying which lock or barrier
implementation a function corresponds to.

```cpp
/* Acquire for POSIX mutex */
void Acquire_pthread_mutex (pthread_mutex_t lock)
{
   pthread_mutex_lock(&lock);
}

/* Release for POSIX mutex */
void Release_pthread_mutex (pthread_mutex_t lock)
{
   pthread_mutex_unlock(&lock);
}

/* Barrier for POSIX */
void Barrier_pthread (pthread_barrier_t barrier)
{
   pthread_barrier_wait(&barrier);
}
```

In another file named pthread_main_lock.x, put your multi-threaded test program for benchmarking
lock designs using the POSIX threads. It is okay if this file has to be manually changed for calling
appropriate Acquire/Release functions. In yet another file named omp_main_lock.x, put your
multi-threaded test program for benchmarking #pragma omp critical. Similarly, create
pthread_main_barrier.x and omp_main_barrier.x.

Put all your results along with observed trends and explanation in a PDF report file.

Please send your submission (five program files and a PDF report) via email to cs433submit2022@gmail.com with subject "Assignment#2 submission Group#X". Replace X by your group number in the subject line.


MAILS
-----

## Mail 1
For implementing the test and set function, you can use the xchg
instruction also if you desire to do so. In fact, the xchg instruction
suites the purpose better than the cmpxchg instruction, but since we did
not discuss any code example in the class using the x86 xchg instruction,
I did not mention it in the assignment. The relevant pages from x86
software development manual are linked with the course page.

## Mail 2
Since in most of your lock implementation, the unlock function is just a
store operation, the compiler is going to inline that function in your
main program. So, now your critical section and the unlock taken together
may look like the following (not correct x86 sequence, but captures the
essence).

```asm
load r, y      // r is some register
inc r          // r++
store r, x     // x <-- r
store r, y     // y <-- r
store 0, lock  // release lock
```

This is okay and in fact, good because it rids you of the overhead of
calling functions. However, since the compiler won't know that the last
store is an unlock, it will see all the three stores as normal stores.
Therefore, it can reorder them arbitrarily because these are independent
stores. Note that Intel CPUs will not do this reordering because they
maintain total store order (TSO), but the compiler can do. For example,
after reordering, the code may look like the following.

```asm
load r, y      // r is some register
inc r          // r++
store r, x     // x <-- r
store 0, lock  // release lock
store r, y     // y <-- r
```

However, such reordering would violate atomicity of the critical section
because some other thread may get the lock even before the previous
critical section's updates are not completed (the assertion x==y will fail
if one update is done, but not the other). To stop the compiler from doing
this reordering, you need to insert an "empty" inline assembly instruction
specifying that this "empty" instruction may change memory contents (i.e.,
place "memory" in clobber list) just before you set (*lock) to zero in
your release function. This forces the compiler to schedule all pending
memory operations before the next instruction (which is the unlock store).
An example release function is shown below. Note that this inline assembly
code does not introduce any extra instruction in the compiled code, but
stops the compiler from doing the incorrect reordering. This is
essentially a hack to locally turn off compiler-induced instruction
reordering.

```cpp
void Release (int *lock)
{
   asm("":::"memory");
   (*lock) = 0;
}
```

Note that this reordering is more likely to happen as you increase the
optimization level of the compiler e.g., -O3 will most probably do this
reordering while -O0 may not.

## Mail 3
In the Acquire function of Lamport's Bakery algorithm, the for loop that
determines who gets to enter the critical section next is built on two
assumptions.

1. If choosing[j] is true for some thread j, this should be made visible
to all threads before thread j proceeds to pick a ticket value.

2. When a thread j picks a new ticket value, this should be made visible
to all threads before thread j enters the for loop.

In Intel CPUs, the store operations may be delayed and subsequent loads to
different addresses are allowed to complete even before preceding stores
are incomplete as per the TSO definition. As a result, the aforementioned
two stores i.e., choosing[i] = true and ticket[i] = max+1 may not become
visible to all threads as expected. In particular, we want choosing[i] =
true to become visible to all threads before thread i proceeds to pick a
ticket value and we want ticket[i]=max+1 to become visible to all threads
before thread i enters the for loop. To make sure that choosing[i] = true
becomes visible before thread i proceeds to pick a ticket, we need to put
a memory fence instruction as discussed in the class. In x86, this can be
done by inserting asm("mfence":::"memory") right after choosing[i] = true.
Similarly, to ensure that ticket[i]=max+1 becomes visible to all threads
before thread i proceeds any further, we need to insert
asm("mfence":::"memory") right after ticket[i]=max+1.