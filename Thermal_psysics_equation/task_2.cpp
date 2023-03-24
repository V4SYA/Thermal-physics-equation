#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>
#include </usr/local/cuda/include/nvtx3/nvToolsExt.h>

double CORNER_1 = 10;
double CORNER_2 = 20;
double CORNER_3 = 30;
double CORNER_4 = 20;


int main(int argc, char **argv){

    int size = 512;
    int iters = 1000000;

    double accur = 10e-6;

    for (int i = 0; i < argc - 1; i++) {
        std::string arg = argv[i];
        if (arg == "-accur") {
            std::string dump = std::string(argv[i + 1]);
            accur = std::stod(dump);
        }
        else if (arg == "-a") size = std::stoi(argv[i + 1]);
        else if (arg == "-i") iters = std::stoi(argv[i + 1]);
    }

    double* A = new double[size * size];
    double* Anew = new double[size * size];

    std::memset(A, 0, sizeof(double) * size * size);

    A[0] = CORNER_1;
    A[size - 1] = CORNER_2;
    A[size * size - 1] = CORNER_3;
    A[size * (size - 1)] = CORNER_4;

    int full_size = size * size;
    double step = (CORNER_2 - CORNER_1) / (size - 1);

    clock_t start = clock();
    for (int i = 1; i < size - 1; i++) {
        A[i] = CORNER_1 + i * step;
        A[i * size + (size - 1)] = CORNER_2 + i * step;
        A[i * size] = CORNER_1 + i * step;
        A[size * (size - 1) + i] = CORNER_4 + i * step;
    }

    std::memcpy(Anew, A, sizeof(double) * full_size);

    clock_t end = clock();

    double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    double err = 1.0;
    int iter = 0;
    double tol = accur;
    int eter_max = iters;
    start = clock();

#pragma acc enter data copyin(Anew[0:full_size], A[0:full_size], err, iter, tol, eter_max)
    while (err > tol && iter < eter_max) {
        iter++;
        if(iter % 100 == 0){
            #pragma acc kernels async(1)
            err = 0.0;
            #pragma acc update device(err) async(1)
        }

        #pragma acc data present(A, Anew, err)
        #pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) reduction(max:err) async(1)
        for (int i = 1; i < size - 1; i++){
            for (int j = 1; j < size - 1; j++){
                Anew[i * size + j] = 0.25 * (A[i * size + j - 1] + A[(i - 1) * size + j] + A[(i + 1) * size + j] + A[i * size + j + 1]);
                err = fmax(err, Anew[i * size + j] - A[i * size + j]);
            }
        }

        if (iter % 100 == 0){
            #pragma acc update host(err) async(1)
            #pragma acc wait(1)
        }

        double* tmp = A;
        A = Anew;
        Anew = tmp;
    }

    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

    std::cout << "Error: " << err << std::endl;
    std::cout << "Iter: " << iter << std::endl;

    free(A);
    free(Anew);

    return 0;
}
