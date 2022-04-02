#include<iostream>
#include<cassert>
#include<pthread.h>
#include<semaphore.h>
using namespace std;

/*
*************************************************
-------------------------------------------------
------------------- LOCK DESIGN -----------------
-------------------------------------------------
*************************************************
*/
#define MFENCE asm volatile ("mfence" ::: "memory")
enum LockType {BAKERY_LOCK = 0, SPIN_LOCK, TTS_LOCK, TICKET_LOCK, ARRAY_LOCK, MUTEX, SEMAP};

// ----------------------------------------------
// ---------- LAMPORT'S BAKERY LOCK -------------
// ----------------------------------------------
const int P_MAX = 128;
struct BakeryLock {
    int dummy[16];
    int P[16];
    int choosing[P_MAX][16];
    int ticket[P_MAX][16];
};

void Init_Bakery(struct BakeryLock* lbl, int P){
    if(P > P_MAX){
        cout<<"Maximum number of threads is 128.";
        exit(1);
    }
    lbl->P[0] = P;
    for(int i = 0; i<lbl->P[0]; i++){
        lbl->choosing[i][0] = 0;
        lbl->ticket[i][0] = 0;
    }
}
void Acquire_Bakery(struct BakeryLock* lbl, int& tid){
    lbl->choosing[tid][0] = 1;
    MFENCE;
    int maxm = -1;
    for(int i = 0; i<lbl->P[0]; i++){
        maxm = max(maxm, lbl->ticket[i][0]);
    }
    lbl->ticket[tid][0] = maxm + 1;
    MFENCE;
    lbl->choosing[tid][0] = 0;

    for(int j = 0; j<lbl->P[0]; j++){
        while(lbl->choosing[j][0]);
        while(
            lbl->ticket[j][0] 
            && 
            ( 
                pair<int, int>{lbl->ticket[j][0], j} 
                < 
                pair<int, int>{lbl->ticket[tid][0], tid} 
            )
        );
    }
}
void Release_Bakery(struct BakeryLock* lbl, int tid){
    MFENCE;
    lbl->ticket[tid][0] = 0;
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- SPIN LOCK -------------------------
// ----------------------------------------------
struct SpinLock {
    int lock;
};
unsigned char CompareAndSet(int oldVal, int newVal, int *ptr) {
    int oldValOut;
    unsigned char result;
    asm("lock cmpxchgl %4, %1 \n setzb %0"
                :"=qm"(result),  "+m" (*ptr), "=a" (oldValOut)
                :"a" (oldVal),  "r" (newVal)
                : );
    return result;
}

// unsigned char CompareAndSet(int oldVal, int newVal, int* ptr){
//     unsigned char result;
//     asm volatile ("lock cmpxchgl %2, %1 \n setzb %0"
//         :"=qm"(result),  "+m" (*ptr)
//         :"r" (newVal)
//         : );
//     return result;
// }

void Init_Spin(struct SpinLock* sl){
    sl->lock = 0;
}
void Acquire_Spin(struct SpinLock* sl){
    while(!CompareAndSet(0, 1, &(sl->lock)));
}
void Release_Spin(struct SpinLock* sl){
    sl->lock = 0;
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- TTS LOCK --------------------------
// ----------------------------------------------
struct TTSLock {
    int lock;
};

void Init_TTS(struct TTSLock* ttsl){
    ttsl->lock = 0;
}

int TestAndSet(int* ptr, int setVal){
    asm volatile (
        "lock xchg %0, %1"
        : "+r"(setVal), "+m"(*ptr)
        : 
        :
    );
    return setVal;
}
void Acquire_TTS(struct TTSLock* ttsl){
    while( 
        ttsl->lock 
        || 
        TestAndSet
        (
            &(ttsl->lock), 
            1
        )
    ){
        continue;
    }
}
void Release_TTS(struct TTSLock* ttsl){
    ttsl->lock = 0;
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- TICKET LOCK -----------------------
// ----------------------------------------------
struct TicketLock {
    int ticket;
    int released;
};
int FetchAndInc(int* ticket){
    int result = 1;
    asm volatile(
        "lock xadd %1, %0"
        : "+m"(*ticket), "+r"(result)
        :
        :
    );
    return result;
}
void Init_Ticket(struct TicketLock* tl){
    tl->ticket = 0;
    tl->released = 0;
}
void Acquire_Ticket(struct TicketLock* tl){
    int ticket = FetchAndInc(&(tl->ticket));
    // MFENCE;
    while(tl->released != ticket);
}
void Release_Ticket(struct TicketLock* tl){
    tl->released++;
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- ARRAY LOCK ------------------------
// ----------------------------------------------
struct ArrayLock {
    int P[16];
    int next[16];
    char avail[P_MAX][64];
};

void Init_Array(struct ArrayLock* al, int num_threads){
    al->P[0] = num_threads;
    al->next[0] = 0;
    for(int i = 0; i<al->P[0]; i++){
        al->avail[i][0] = 0;
    }
    al->avail[0][0] = 1;
}
void Acquire_Array(struct ArrayLock* al, int& tid){
    tid = FetchAndInc(&(al->next[0]))%(al->P[0]);
    while(al->avail[tid][0] != 1);

} 
void Release_Array(struct ArrayLock* al, int& tid){
    al->avail[tid][0] = 0;
    al->avail[(tid+1)%(al->P[0])][0] = 1;
}

// ----------------------------------------------

// ----------------------------------------------
// ---------- PTHREAD_MUTEX ---------------------
// ----------------------------------------------

struct MutexLock {
    pthread_mutex_t lock;
};

void Init_Mutex(struct MutexLock * ml){
    pthread_mutex_init(&(ml->lock), NULL);
}
void Acquire_Mutex(struct MutexLock * ml){
    pthread_mutex_lock(&(ml->lock));
}
void Release_Mutex(struct MutexLock * ml){
    pthread_mutex_unlock(&(ml->lock));
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- SEMAPHORE -------------------------
// ----------------------------------------------
struct SemLock {
    sem_t sem;
};
void Init_Sem(struct SemLock* sl){
    sem_init(&(sl->sem), 0, 1);
}
void Acquire_Sem(struct SemLock* sl){
    sem_wait(&(sl->sem));
}
void Release_Sem(struct SemLock* sl){
    sem_post(&(sl->sem));
}
// ----------------------------------------------

// ---------- WRAPPER STRUCT FOR LOCK -----------
struct Lock {
    int lock_type[16];
    void* lock[16];
};

// ---------- GENERAL FUNCTION FOR LOCK_INIT ---
struct Lock* Init(int lock_type, int num_threads){
    void* core_lock = NULL;
    switch (lock_type){
        case BAKERY_LOCK: {
            struct BakeryLock *lbl = (struct BakeryLock*)malloc(sizeof(BakeryLock));
            Init_Bakery(lbl, num_threads);
            core_lock = (void*)lbl;
            break;
        }
        case SPIN_LOCK: {
            struct SpinLock* sl = (struct SpinLock*)(malloc(sizeof(SpinLock)));
            Init_Spin(sl);
            core_lock = (void*)sl;
            break;
        }
        case TTS_LOCK: {
            struct TTSLock* ttsl = (struct TTSLock*)(malloc(sizeof(TTSLock)));
            Init_TTS(ttsl);
            core_lock = (void*)ttsl;
            break;
        }
        case TICKET_LOCK: {
            struct TicketLock* tl = (struct TicketLock*)(malloc(sizeof(TicketLock)));
            Init_Ticket(tl);
            core_lock = (void*)tl;
            break;
        }
        case ARRAY_LOCK: {
            struct ArrayLock* al = (struct ArrayLock*)(malloc(sizeof(ArrayLock)));
            Init_Array(al, num_threads);
            core_lock = (void*)al;
            break;
        }
        case MUTEX: {
            struct MutexLock* ml = (struct MutexLock*)(malloc(sizeof(MutexLock)));
            Init_Mutex(ml);
            core_lock = (void*)ml;
            break;
        }
        case SEMAP: {
            struct SemLock* sl = (struct SemLock*)(malloc(sizeof(SemLock)));
            Init_Sem(sl);
            core_lock = (void*)sl;
            break;
        }
        default: {
            cout<<"Unsupported Lock Type\n";
        }
    }
    struct Lock * lock = (struct Lock*)malloc(sizeof(Lock));
    lock->lock_type[0] = lock_type;
    lock->lock[0] = core_lock;
    return lock;
}

// ---------- GENERAL FUNCTION FOR LOCK_ACQUIRE --
void Acquire(struct Lock* lock, int& tid){
    switch (lock->lock_type[0]){ 
        case BAKERY_LOCK: {
            Acquire_Bakery((struct BakeryLock*)(lock->lock[0]), tid);
            break;
        }
        case SPIN_LOCK: {
            Acquire_Spin((struct SpinLock*)lock->lock[0]);
            break;
        }
        case TTS_LOCK: {
            Acquire_TTS((struct TTSLock*)lock->lock[0]);
            break;
        }
        case TICKET_LOCK: {
            Acquire_Ticket((struct TicketLock*)lock->lock[0]);
            break;
        }
        case ARRAY_LOCK: {
            Acquire_Array((struct ArrayLock* )lock->lock[0], tid);
            break;
        }
        case MUTEX: {
            Acquire_Mutex((struct MutexLock*)lock->lock[0]);
            break;
        }
        case SEMAP: {
            Acquire_Sem((struct SemLock*)lock->lock[0]);
            break;
        }
        default: {
            cout<<"Unsupported Lock Type\n";
            exit(1);
        }
    }
}

// ---------- GENERAL FUNCTION FOR LOCK_RELEASE --
void Release(struct Lock* lock, int& tid){
    switch (lock->lock_type[0]){ 
        case BAKERY_LOCK: {
            Release_Bakery((struct BakeryLock*)(lock->lock[0]), tid);
            break;
        }
        case SPIN_LOCK: {
            Release_Spin((struct SpinLock*)lock->lock[0]);
            break;
        }
        case TTS_LOCK: {
            Release_TTS((struct TTSLock*)lock->lock[0]);
            break;
        }
        case TICKET_LOCK: {
            Release_Ticket((struct TicketLock*)lock->lock[0]);
            break;
        }
        case ARRAY_LOCK: {
            Release_Array((struct ArrayLock* )lock->lock[0], tid);
            break;
        }
        case MUTEX: {
            Release_Mutex((struct MutexLock*)lock->lock[0]);
            break;
        }
        case SEMAP: {
            Release_Sem((struct SemLock*)lock->lock[0]);
            break;
        }
        default: {
            cout<<"Unsupported Lock Type\n";
            exit(1);
        }
    }
}

// ---------- GENERAL FUNCTION FOR FREEING LOCK --
void Free(struct Lock* lock){
    switch (lock->lock_type[0]){ 
        case BAKERY_LOCK: {
            free((struct BakeryLock*)(lock->lock[0]));
            break;
        }
        case SPIN_LOCK: {
            free((struct SpinLock*)lock->lock[0]);
            break;
        }
        case TTS_LOCK: {
            free((struct TTSLock*)lock->lock[0]);
            break;
        }
        case TICKET_LOCK: {
            free((struct TicketLock*)lock->lock[0]);
            break;
        }
        case ARRAY_LOCK: {
            free((struct ArrayLock* )lock->lock[0]);
            break;
        }
        case MUTEX: {
            pthread_mutex_destroy(&((struct MutexLock*)lock->lock[0])->lock);
            free((struct MutexLock*)lock->lock[0]);
            break;
        }
        case SEMAP: {
            sem_destroy(&((struct SemLock*)lock->lock[0])->sem);
            free((struct SemLock*)lock->lock[0]);
            break;
        }
        default: {
            cout<<"Unsupported Lock Type\n";
            exit(1);
        }
    }
    free(lock);
}

/*
*************************************************
-------------------------------------------------
------------------- BARRIER DESIGN --------------
-------------------------------------------------
*************************************************
*/

enum BarrierType {CENTRAL_BUSY=0, TREE_BUSY, CENTRAL_CONDV, TREE_CONDV, POSIX_BARRIER};

// ----------------------------------------------
// ---------- CENTRAL_BUSY ----------------------
// ----------------------------------------------
struct CentralBusy {
    int P;
    int ctr;
    int flag;
    pthread_mutex_t lock;
};
void Init_CentralBusy(struct CentralBusy* cb, int num_threads) {
    cb->flag = 0;
    pthread_mutex_init(&(cb->lock), NULL);
    cb->ctr = 0;
    cb->P = num_threads;
}
void Set_CentralBusy(struct CentralBusy * cb, int* sense){
    *sense = 1 - *sense;
    pthread_mutex_lock(&(cb->lock));
    cb->ctr++;
    if(cb->ctr == cb->P){
        pthread_mutex_unlock(&(cb->lock));
        cb->ctr = 0;
        cb->flag = *sense;
    } else{
        pthread_mutex_unlock(&(cb->lock));
        while(cb->flag != *sense);
    }
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- TREE_BUSY ----------------------
// ----------------------------------------------
struct TreeBusy {
    int P;
    int flag[P_MAX][16];
};
void Init_TreeBusy(struct TreeBusy * tb, int num_threads){
    tb->P = num_threads;
    for(int i = 0; i<tb->P; i++){
        for(int j = 0; j<16; j++){
            tb->flag[i][j] = 0;
        }
    }
}
void Set_TreeBusy(struct TreeBusy * tb, int tid){
    unsigned int i, mask;
    for(i =0, mask = 1; (mask & tid) !=0; ++i, mask <<= 1){
        while(!tb->flag[tid][i]);
        tb->flag[tid][i] = 0;
    }
    if(tid < tb->P - 1){
        tb->flag[tid + mask][i] = 1;
        while(!tb->flag[tid][15]);
        tb->flag[tid][15] = 0;
    }
    for(mask >>=1; mask > 0; mask>>=1) {
        tb->flag[tid - mask][15] = 1;
    }
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- CENTRAL_CONDV ---------------------
// ----------------------------------------------

struct CentralCondv{
    int ctr;
    pthread_mutex_t lock;
    pthread_cond_t cv;
    int P;
};

void Init_CentralCondv(struct CentralCondv* cc, int num_threads){
    pthread_mutex_init(&(cc->lock), NULL);
    pthread_cond_init(&(cc->cv), NULL);
    cc->ctr = 0;
    cc->P = num_threads;
}
void Set_CentralCondv(struct CentralCondv* cc){
    pthread_mutex_lock(&(cc->lock));
    cc->ctr++;
    if(cc->ctr == cc->P){
        cc->ctr = 0;
        pthread_cond_broadcast(&(cc->cv));
    } else{
        while(pthread_cond_wait(&(cc->cv), &(cc->lock)) != 0);
    }
    pthread_mutex_unlock(&(cc->lock));
}

// ----------------------------------------------

// ----------------------------------------------
// ---------- TREE_CONDV ------------------------
// ----------------------------------------------
struct TreeNodeCondv{
    int flag[16][16];
    pthread_cond_t cv[16][16];
    pthread_mutex_t lock[16][16];
};
struct TreeCondv {
    struct TreeNodeCondv nodes[P_MAX];
    int P;
};

void Init_TreeCondv(struct TreeCondv * tc, int num_threads){
    tc->P = num_threads;
    for(int i = 0; i<tc->P; i++){
        for(int j = 0; j<16; j++){
            tc->nodes[i].flag[j][0] = 0;
            pthread_mutex_init(&(tc->nodes[i].lock[j][0]), NULL);
            pthread_cond_init(&(tc->nodes[i].cv[j][0]), NULL); 
        }
    }
}
void Set_TreeCondv(struct TreeCondv* tc, int tid){
    unsigned int i, mask;
    for(i =0, mask = 1; (mask & tid) !=0; ++i, mask <<= 1){
        pthread_mutex_lock(&(tc->nodes[tid].lock[i][0]));
        while(!tc->nodes[tid].flag[i][0]){
            pthread_cond_wait(&(tc->nodes[tid].cv[i][0]), &(tc->nodes[tid].lock[i][0]));
        }
        tc->nodes[tid].flag[i][0] = 0;
        pthread_mutex_unlock(&(tc->nodes[tid].lock[i][0]));
    }
    if(tid < tc->P - 1){
        // if(!FetchAndInc(&tc->nodes[tid+mask].flag[i][0])){
        //     pthread_cond_signal(&(tc->nodes[tid+mask].cv[i][0]));
        // }
        if(!tc->nodes[tid+mask].flag[i][0]){
            pthread_mutex_lock(&(tc->nodes[tid+mask].lock[i][0]));
            tc->nodes[tid+mask].flag[i][0] = 1;
            pthread_cond_signal(&(tc->nodes[tid+mask].cv[i][0]));
            pthread_mutex_unlock(&(tc->nodes[tid+mask].lock[i][0]));
        }
        pthread_mutex_lock(&(tc->nodes[tid].lock[15][0]));
        while(!tc->nodes[tid].flag[15][0]){
            pthread_cond_wait(&(tc->nodes[tid].cv[15][0]), &(tc->nodes[tid].lock[15][0]));
        }
        tc->nodes[tid].flag[15][0] = 0;
        pthread_mutex_unlock(&(tc->nodes[tid].lock[15][0]));
    }
    for(mask >>=1; mask > 0; mask>>=1) {
        if(!tc->nodes[tid-mask].flag[15][0]){
            pthread_mutex_lock(&(tc->nodes[tid-mask].lock[15][0]));
            tc->nodes[tid-mask].flag[15][0] = 1;
            pthread_mutex_unlock(&(tc->nodes[tid-mask].lock[15][0]));
            pthread_cond_signal(&(tc->nodes[tid-mask].cv[15][0]));
        }
    }
}
// ----------------------------------------------

// ----------------------------------------------
// ---------- POSIX_BARRIER ---------------------
// ----------------------------------------------

struct PosixBarrier {
    pthread_barrier_t barrier;
};

void Init_Posix(struct PosixBarrier* pb){
    pthread_barrier_init(&(pb->barrier), NULL, 0);
}

void Set_Posix(struct PosixBarrier* pb){
    pthread_barrier_wait(&(pb->barrier));
}
// ----------------------------------------------

struct Barrier {
    int barrier_type[16];
    void* barrier[16];
};

struct Barrier* Init_Barrier(int barrier_type, int num_threads){
    void * core_barrier = NULL;
    switch (barrier_type) {
        case CENTRAL_BUSY: {
            struct CentralBusy* cb = (struct CentralBusy*)(malloc(sizeof(CentralBusy)));
            Init_CentralBusy(cb, num_threads);
            core_barrier = (void*)cb;
            break;
        }
        case TREE_BUSY: {
            struct TreeBusy* tb = (struct TreeBusy*)(malloc(sizeof(TreeBusy)));
            Init_TreeBusy(tb, num_threads);
            core_barrier = (void*)tb;
            break;
        }
        case CENTRAL_CONDV: {
            struct CentralCondv* cc = (struct CentralCondv*)(malloc(sizeof(CentralCondv)));
            Init_CentralCondv(cc, num_threads);
            core_barrier = (void*)cc;
            break;
        }
        case TREE_CONDV: {
            struct TreeCondv* tc = (struct TreeCondv*)(malloc(sizeof(TreeCondv)));
            Init_TreeCondv(tc, num_threads);
            core_barrier = (void*)tc;
            break;
        }
        case POSIX_BARRIER: {
            struct PosixBarrier* pb = (struct PosixBarrier*)(malloc(sizeof(PosixBarrier)));
            Init_Posix(pb);
            core_barrier = (void*)pb;
            break;
        }
    }
    struct Barrier * barrier = (struct Barrier*)malloc(sizeof(Barrier));
    barrier->barrier_type[0] = barrier_type;
    barrier->barrier[0] = core_barrier;
    return barrier;
}

void Set_Barrier(struct Barrier* bar, int * sense, int tid){
    switch (bar->barrier_type[0]) {
        case CENTRAL_BUSY: {
            Set_CentralBusy((struct CentralBusy*)bar->barrier[0], sense);
            break;
        }
        case TREE_BUSY: {
            Set_TreeBusy((struct TreeBusy*)bar->barrier[0], tid);
            break;
        }
        case CENTRAL_CONDV: {
            Set_CentralCondv((struct CentralCondv*)bar->barrier[0]);
            break;
        }
        case TREE_CONDV: {
            Set_TreeCondv((struct TreeCondv*)bar->barrier[0], tid);
            break;
        }
        case POSIX_BARRIER: {
            Set_Posix((struct PosixBarrier*)bar->barrier[0]);
            break;
        }
    }
}

void Free_Barrier(struct Barrier* bar){
    switch (bar->barrier_type[0]) {
        case CENTRAL_BUSY: {
            pthread_barrier_destroy(&(((struct CentralBusy*)(bar->barrier[0]))->lock));
            free((struct CentralBusy *)bar->barrier[0]);
            break;
        }
        case TREE_BUSY: {
            free((struct TreeBusy*)bar->barrier[0]);
            break;
        }
        case CENTRAL_CONDV: {
            pthread_barrier_destroy(&(((struct CentralCondv*)(bar->barrier[0]))->lock));
            pthread_cond_destroy(&(((struct CentralCondv*)(bar->barrier[0]))->cv));
            free((struct CentralCondv*)bar->barrier[0]);
            break;
        }
        case TREE_CONDV: {
            struct TreeCondv* tc = (struct TreeCondv*)(bar->barrier[0]);
            for(int i = 0; i<tc->P; i++){
                for(int j = 0; j<16; j++){
                    pthread_mutex_destroy(&(tc->nodes[i].lock[j][0]));
                    pthread_cond_destroy(&(tc->nodes[i].cv[j][0]));
                }
            }
            free(tc);
            break;
        }
        case POSIX_BARRIER: {
            pthread_barrier_destroy(&((struct PosixBarrier* )(bar->barrier[0]))->barrier);
            free((struct PosixBarrier *)bar->barrier[0]);
            break;
        }
    }
    free(bar);
}