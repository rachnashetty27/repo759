//this task has been done with the help of claude.ai and google search
#include <iostream>
#include <cstdio>

int main(int argc, char* argv[]) {
    // Check if argument is provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    // Convert command line argument to integer
    int N = std::atoi(argv[1]);

    // Print ascending order using printf
    for (int i = 0; i <= N; i++) {
        printf("%d ", i);
    }
    printf("\n");

    // Print descending order using std::cout
    for (int i = N; i >= 0; i--) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}