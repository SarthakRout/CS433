#include<iostream>
#include<fstream>
#include<vector>
#include<iomanip>
#include<chrono>
#include<stack>

using namespace std;
#define ll long long int
const ll INF = 1LL<<17; 

// https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
// Function to generate the next permutation of state having same number of set bits
ll NextPerm(ll msk){
    ll z = msk | (msk - 1);
    return (z + 1) | (((~z & -~z) - 1) >> (__builtin_ctzll(msk) + 1));
}

// Global counters to record time taken
double solve_time_taken;
int nn;

void Solve(int num_threads){

    // Taking in size of the graph as input
    int n;
    std::cin>>n;
    nn = n;
    
    // Initialising the distance array to store the edge weights
    int dist[n][n];
    for(int i = 0; i<n; i++){
        dist[i][i] = 0;
        for(int j = i+1; j<n; j++){
            std::cin>>dist[i][j];
            dist[j][i] = dist[i][j]; // Symmetric TSP
        }
    }

    // Container of dimensions 2^n x n to store cost of subproblems
    //  initialised with a very large value INF
    vector<vector<ll>> cost((1LL<<n)+2, vector<ll>(n, INF));
    auto solve_start = std::chrono::high_resolution_clock::now(); // Start measuring time for the algorithm
    cost[1][0] = 0;
    ll msk, state;

    // https://stackoverflow.com/questions/27653672/tsp-using-dynamic-programming
    // Iterating over subset sizes from 0 to n-1 for all subsets excluding the source vertex
    for(int sz=0; sz<n; sz++){
        // First mask 000...00111..11 indicating the first subset
        msk = (1LL<<sz)-1;
        vector<ll> states; // To store all subsets of the size sz to be used by the threads later
        while(1){
            // The state with source vertex appended is pushed back to the states vector
            states.push_back((msk<<1)|1);
            // Generate the next permutation if we have not reached the last state of the form 11..1100..00
            if(msk != ((1LL<<sz)-1)<<(n-1-sz)){
                msk = NextPerm(msk);
            }
            else{
                break;
            }
        }
        // OMP directive to parallelise the outer for loop with a default static schedule with num_threads threads
        // Each thread works on states.size()/num_threads states and for each state O(n^2) updates.
        #pragma omp parallel for num_threads(num_threads)
        for(int i = 0;  i<states.size(); i++){
            // states[i] is the current state
            for(int j = 0; j<n; j++){
                if(cost[states[i]][j] == INF){
                    continue;
                }
                // vertex j is present in the subset and we wish to extend from this vertex
                for(int k=1; k<n; k++){
                    if(states[i] & (1LL<<k)){
                        continue;
                    }
                    // Vertex j is not present in the subset and we wish to add it to this subset
                    // Minimise the cost if this edge is chosen for this subset
                    cost[(states[i] | (1LL << k))][k] = min(cost[(states[i] | (1LL << k))][k], cost[states[i]][j] + dist[j][k]);
                }
            }
        }
    }
    // Get the list of vertices that make the tour
    stack<int> tour;
    tour.push(0);
    int last = 0;
    msk = (1LL<<n) - 1;
    for(int i = 1; i<n; i++){
        // For the ith vertex to be visited
        int minid = -1, minm = INF;
        for(int j = 1; j<n; j++){
            // Check if choosing j to be ith vertex minimises the cost
            if((msk && (1LL<<j)) && cost[msk][j] + dist[j][last] < minm){
                minm = cost[msk][j] + dist[j][last];
                minid = j;
            }
        }
        // Record the cost minimising vertex on the stack
        msk = msk - (1LL<<minid);
        last = minid;
        tour.push(last);
    }
    tour.push(0);
    auto solve_end = std::chrono::high_resolution_clock::now(); // Marking end of the algorithm

    // Output the list of vertices that make the tour 
    // and record the last vertex to find the minimum cost
    int min_cost_id = -1;
    while(!tour.empty()){
        if(tour.top()){
            min_cost_id = tour.top();
        }
        std::cout<<tour.top()+1<<" ";
        tour.pop();
    }
    std::cout<<"\n";
    std::cout<<cost[(1LL<<n)-1][min_cost_id] + dist[min_cost_id][0]<<"\n";

    #ifdef SARTHAK
    solve_time_taken =  std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); solve_time_taken *= 1e-9; 
    std::cout << "\nTime taken by TSP operations is : " << fixed << solve_time_taken << setprecision(9); std::cout << " sec" << endl;
    #endif
}
int main(int argc, char** argv){

    // Parsing Arguments
    if(argc != 4){
        std::cout<<"Need exactly 3 arguments: input file name, output file name and number of threads in that order.\n";
        exit(1);
    }
    std::string input_file_name = argv[1];
    std::string output_file_name = argv[2];
    int num_threads = atoi(argv[3]);
    if(freopen( input_file_name.c_str(), "r", stdin) == NULL){
        std::cout<<"Input file name path is invalid\n";
        exit(1);
    }
    if(freopen( output_file_name.c_str(), "w", stdout) == NULL){
        std::cout<<"Output file name path is invalid\n";
        exit(1);
    }
    if(num_threads < 1){
        std::cout<<"Enter an positive integer for number of threads\n";
        exit(1);
    }

    // Fast Input/Output
    ios_base::sync_with_stdio(false);
    std::cin.tie(NULL); std::cout.tie(NULL);
    
    Solve(num_threads);

    #ifdef SARTHAK
    fstream rec;
    rec.open("records1.txt", ios::app);
    rec<<nn<<","<<num_threads<<","<<solve_time_taken<<"\n";
    rec.close();
    #endif
    return 0;
}
