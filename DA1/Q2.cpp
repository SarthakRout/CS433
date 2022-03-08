#include<iostream>
#include<vector>
#include<fstream>
#include<cmath>
#include<chrono>
#include<iomanip>

using namespace std;
#define ll long long int

void InitializeInput(int n, vector<vector<float>> &L, vector<float>&y){
    
    // Parallelise the zeroing of the L matrix
    #pragma omp parallel for num_threads(8)
    for(int i =0 ; i<=n; i++){
        for(int j = 0; j<=n; j++){
            L[i][j] = 0;
        }
    }

    // Parallelise the generation of lower triangular matrix L
    #pragma omp parallel for num_threads(8)
    for(int i = 1; i<=n; i++){
        for(int j = 1; j<i; j++){
            L[i][j] = (1.0+j+i)/(j*i + 4);
        }
        L[i][i] = (1.0*i)/(i + 7);
    }
    // Initialise the y values
    for(int i = 0; i<n; i++){
        y[i] = (1.0*i+10)/(i*2 + 5) + 3.25/(i+1);
    }

}

void ReadInput(int n, vector<vector<float>> &L, vector<float>&y){

    // Parallelise the zeroing of the L matrix
    #pragma omp parallel for num_threads(8)
    for(int i =0 ; i<=n; i++){
        for(int j = 0; j<=n; j++){
            L[i][j] = 0;
        }
    }
    
    // Read the values for the L matrix
    for(int i = 1; i<=n; i++){
        for(int j = 1; j<=i-1; j++){
            std::cin>>L[i][j];
        }
        std::cin>>L[i][i];
    }

    // Read the values of the y vector
    for(int i = 0; i<n; i++){
        std::cin>>y[i];
    }
}

// Global counters for recording the time taken
float solve_time_taken = 0;
ll nn, blksz;

void Solve(int num_threads){

    ll n = (1<<14); // Value of n for local testing
    #ifndef SARTHAK
    std::cin>>n;         // Value of n to be read if working on actual input
    #endif
    nn = n;

    // Containers for matrix L of dimensions n+1 X n+1 and vector y 
    vector<vector<float>> L(n+1, vector<float>(n+1));
    vector<float> y(n);
    // Array for solution vector x
    float x[n];
    
    // Read actual input
    #ifndef SARTHAK
    ReadInput(n, L, y);
    #endif
    // or initialise with locally generated values to L and y
    #ifdef SARTHAK
    InitializeInput(n, L, y);
    #endif
    
    // Block size O(sqrt(n)) for best performance 
    ll block_size = 1<<(int)(log2(1+n*2)/2);
    blksz = block_size;
    ll block_start, block_end;
    // Marking start of algorithm 
    auto solve_start = std::chrono::high_resolution_clock::now(); 
    // Iterating over each block; n/block_size iterations
    for(int b = 0; b<=n/block_size; b++){

        // Finding the start and the end for each block
        block_start = 1 + (block_size)*b;
        block_end = min(n, 1LL*(b+1)*block_size);

        // Solving for each lower triangular block with a single thread
        for(int i = block_start; i<= block_end; i++){
            for(int j = block_start; j<i; j++){
                y[i-1] -= L[i][j]*x[j-1];
            }
            x[i-1] = (y[i-1])/(L[i][i]);
        }

        // Using the recently solved values of x to solve the submatrix of size block_size*(n - b*block_size)
        // Parallelising the above with num_thread threads with default static scheduling
        #pragma omp parallel for num_threads(num_threads)
        for(int i = block_end+1; i<=n; i++){
            float dif = 0;
            for(int j = block_start; j<=block_end; j++){
                dif += L[i][j]*x[j-1];
            }
            y[i-1] -= dif;
        }
    }
    // Marking end of algorithm
    auto solve_end = std::chrono::high_resolution_clock::now(); 

    // Printing the solution; values of x
    for(int i = 0; i<n; i++){
        std::cout<<x[i]<<" ";
    }

    #ifdef SARTHAK
    solve_time_taken =  std::chrono::duration_cast<std::chrono::nanoseconds>(solve_end - solve_start).count(); solve_time_taken *= 1e-9; 
    std::cout << "\nTime taken by matrix operations is : " << fixed << solve_time_taken << setprecision(9); std::cout << " sec" << endl; 
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

    // Fast Input Output
    ios_base::sync_with_stdio(false);
    std::cin.tie(NULL); std::cout.tie(NULL);
    
    Solve(num_threads);

    #ifdef SARTHAK
    fstream rec;
    rec.open("records2.txt", ios::app);
    rec<<nn<<","<<num_threads<<","<<blksz<<","<<solve_time_taken<<",0\n";
    rec.close();
    #endif
    return 0;
}
