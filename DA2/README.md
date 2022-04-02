# CS433 - Second Design Assignment
To run the lock program using POSIX Thread Interface, run

- On Windows
```bash
g++ .\pthread_main_lock.cpp .\sync_library.hpp -o pthread_main_lock; 
.\pthread_main_lock
```

- On Linux
```bash
g++ ./pthread_main_lock.cpp ./sync_library.hpp -o pthread_main_lock; 
./pthread_main_lock
```