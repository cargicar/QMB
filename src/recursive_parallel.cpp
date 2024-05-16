#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
   
using namespace std;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;

const int n = 14;

struct Array2D {
    double arr[n][n];
    bool non_phys;
};

void print_time( high_resolution_clock::time_point startTime,
                   high_resolution_clock::time_point endTime) {
  printf("Time: %fms\n",
         duration_cast<duration<double, milli>>(endTime - startTime).count());
}

double xs(int s, double E, double g, double w, double x2) {
    if (s % 2 == 1 || s < 0) {
        return 0;
    } else if (s == 0) {
        return 1;
    } else if (s == 2) {
        return x2;
    } else {
        return (4 * (s - 3) * E * xs(s - 4, E, g, w, x2) + (s - 3) * (s - 4) * (s - 5) * xs(s - 6, E, g, w, x2) - 4 * w * (s - 2) * xs(s - 2, E, g, w, x2)) / (4 * g * (s - 1));
    }
}

Array2D getHankel(double e, double g, double w, double x2) {
    Array2D var;
    /// int n = ...; // assume this is defined elsewhere
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          double hij  = xs(i + j, e, g, w, x2);
	  var.arr[i][j]=hij;
        }
    }
    return var;
}


Array2D getCholesky(double e, double g, double w, double x2) {
    Array2D var;
    Array2D hankel = getHankel(e, g, w, x2);

   
    for (int j = 0; j < n; ++j) {
        double sum = 0;
	volatile bool flag=false;
        #pragma omp parallel for reduction(+:sum)  
        for (int k = 0; k < j; ++k) {
            sum += var.arr[j][k] * var.arr[j][k];
        }
        var.arr[j][j] = sqrt(hankel.arr[j][j] - sum);
        #pragma omp parallel for reduction(+:sum)	
        for (int i = j + 1; i < n; ++i) {
            sum = 0;
	    if(flag) continue;
            for (int k = 0; k < j; ++k) {
                sum += var.arr[i][k] * var.arr[j][k];
            }
            var.arr[i][j] = (1.0 / var.arr[j][j] * (hankel.arr[i][j] - sum));
            var.non_phys = isnan(var.arr[i][j]);
            if (var.non_phys) {
                flag=true;
            }
        }
    }
    return var;
}

// Array2D getCholesky(double e, double g, double w, double x2) {
//     Array2D var;
//     Array2D hankel = getHankel(e, g, w, x2);

//     for (int j = 0; j < n; ++j) {
//         double sum = 0;
//         for (int k = 0; k < j; ++k) {
//             sum += var.arr[j][k] * var.arr[j][k];
//         }
//         var.arr[j][j] = sqrt(hankel.arr[j][j] - sum);

//         for (int i = j + 1; i < n; ++i) {
//             sum = 0;
//             for (int k = 0; k < j; ++k) {
//                 sum += var.arr[i][k] * var.arr[j][k];
//             }
//             var.arr[i][j] = (1.0 / var.arr[j][j] * (hankel.arr[i][j] - sum));
//             var.non_phys = isnan(var.arr[i][j]);
//             if (var.non_phys) {
//                 break;
//             }
//         }
//     }
//     return var;
// }


void printArray(Array2D var) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << var.arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int xsize;
    double xlower, xupper, elower, eupper, g = 1.0, w = -5.0;

    std::cout << "Enter the size of the x-grid: ";
    std::cin >> xsize;

    std::cout << "Enter the lower limit for x-region: ";
    std::cin >> xlower;
    std::cout << "Enter the upper limit for x-region: ";
    std::cin >> xupper;
    std::cout << "Enter the lower limit for e-region: ";
    std::cin >> elower;
    std::cout << "Enter the upper limit for e-region: ";
    std::cin >> eupper;

    double xinter = xupper - xlower;
    double einter = eupper - elower;
    
    std::ofstream myfile("island_parallel.txt");
    std::ofstream gridfile("grid_parallel.txt");
   
    gridfile << n << " " << xsize << std::endl;
    const auto startTime = high_resolution_clock::now();

// #pragma omp parallel for collapse(2)
//     for (int i = 0; i < xsize; ++i) {
//     double xss = xlower + i * xinter / xsize;
//     for (int j = 0; j < xsize; ++j) {
//         double ess = elower + j * einter / xsize;
//         Array2D L = getCholesky(ess, g, w, xss);
//         if (!L.non_phys && myfile.is_open()) {
//             #pragma omp critical
//             myfile << xss << " " << ess << std::endl;
//         }
//     }
//     }
    
    for (int i = 0; i < xsize; ++i) {
        double xss = xlower + i * xinter / xsize;
        for (int j = 0; j < xsize; ++j) {
            double ess = elower + j * einter / xsize;
            Array2D L = getCholesky(ess, g, w, xss);
            if (!L.non_phys && myfile.is_open()) {
	      myfile << xss << " " << ess << std::endl;
            }
        }
    }
  
    myfile.close();
    gridfile.close();
    const auto endTime = high_resolution_clock::now();
    print_time(startTime, endTime);
    return 0;
}
