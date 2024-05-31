/* Carlos Cardona-Giraldo 2024 */
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
         duration_cast<duration<double, milli> >(endTime - startTime).count());
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

Array2D getHankel(double e, double  g, double w, double x2) {
    Array2D var;
    double hij;
    int i, j;
    
    ///#pragma omp parallel for shared(hij, var)  private(i, j)

    for ( i = 0; i < n; ++i) {
        for ( j = 0; j < n; ++j) {
          hij  = xs(i + j, e, g, w, x2);
	  var.arr[i][j]=hij;
	// nth = omp_get_thread_num();
        // std::cout << "num of threats:  "<< nth<<  std::endl;
        }
    }
    return var;
}


Array2D getCholesky(double e, double g, double w, double x2) {
    Array2D var;
    Array2D hankel = getHankel(e, g, w, x2);
    int i, j, k;
    double sum; 
    ///#pragma omp parallel for reduction(+:sum)
    for ( j = 0; j < n; ++j) {
        sum = 0;
	volatile bool flag=false;
        for (int k = 0; k < j; ++k) {
            sum += var.arr[j][k] * var.arr[j][k];
        }
        var.arr[j][j] = sqrt(hankel.arr[j][j] - sum);
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


void printArray(Array2D var) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << var.arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
  Array2D L;
  int xsize, i, j;
  double xss, ess, xlower, xupper, elower, eupper, g = 1.0, w = -5.0;
 
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
    int xs2 = 5000000; /// max xsize, fixed before compiling to avoid dynamic alloc.
    double* ees = new double[xs2];
    double* xxs = new double[xs2];
   
    const auto startTime = high_resolution_clock::now();
    
#pragma omp parallel private(xss, ess, L) /// shared(xss, ess, ees,xxs, L,i, j)  private(i, j)
    for (i = 0; i < xsize; ++i) {
        xss = xlower + i * xinter / xsize;
        for (j = 0; j < xsize; ++j) {
	    ess = elower + j * einter / xsize;
            L = getCholesky(ess, g, w, xss);
            if (!L.non_phys) {
	       ees[i*xsize+j]=ess;
	       xxs[i*xsize+j]=xss;  
	        // std::cout<<"ess: "<<ess<<std::endl;
	        // std::cout<<"ees array: "<<ees[i+j]<<std::endl;
            }  
	}
    }
   
    std::ofstream xsfile("xs.txt");
    std::ofstream esfile("es.txt");
    std::ofstream gridfile("grid.txt");
   
    gridfile << n << " " << xsize << std::endl;
    /// #pragma omp barrier
    ///#pragma omp single
   
    for (i=0; i < xsize; ++i){
        for (j = 0; j < xsize; ++j) {
	  xsfile<< xxs[i*xsize+j] <<" "<< std::endl;
	  esfile<< ees[i*xsize+j] <<" "<< std::endl;
	}
    }
    xsfile.close();
    esfile.close();
    gridfile.close();
    delete[] ees;
    delete[] xxs; 
    const auto endTime = high_resolution_clock::now();
    print_time(startTime, endTime);
    return 0;
}
