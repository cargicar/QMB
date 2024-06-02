/* Carlos Cardona-Giraldo 2024 */
#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include <vector>

//#include "include/cx.h"

using namespace std;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;

const double g = 1;
const double w = -5;

void print_time( high_resolution_clock::time_point startTime,
                   high_resolution_clock::time_point endTime) {
  printf("Time: %fms\n",
         duration_cast<duration<double, milli> >(endTime - startTime).count());
}

double xs(int s, double E, double x2) {
    if (s % 2 == 1 || s < 0) {
        return 0;
    } else if (s == 0) {
        return 1;
    } else if (s == 2) {
        return x2;
    } else {
        return (4 * (s - 3) * E * xs(s - 4, E, x2) + (s - 3) * (s - 4) * (s - 5) * xs(s - 6, E, x2) - 4 * w * (s - 2) * xs(s - 2, E, x2)) / (4 * g * (s - 1));
    }
}

bool getCholesky(double e, double x2, const int N)
{
  double *hij = (double *)malloc(N * N * sizeof(double));
  double *l = (double *)malloc(N * N * sizeof(double));
  double sum;
  bool non_phys;
  volatile bool flag=false; 
  //fill hankel
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
	hij[i*N+j]  = xs(i + j, e, x2);
        }
    }

   for (int j = 0; j < N; ++j) {
        sum = 0;

        for (int k = 0; k < j; ++k) {
            sum += l[j*N+k] * l[j*N+k];
        }
        l[j*N+j] = sqrt(hij[j*N+j] - sum);
        for (int i = j + 1; i < N; ++i) {
	  sum = 0;
	  if(flag) continue;
            for (int k = 0; k < j; ++k) {
                sum += l[i*N+k] * l[j*N+k];
            }
            l[i*N+j] = (1.0 / l[j*N+j] * (hij[i*N+j] - sum));
	    //	    std::cout<<"lss[i,j]: "<< l[i*N+j]<<std::endl;
            non_phys = isnan(l[i*N+j]);
	    //std::cout<<"non_phys: "<<non_phys<<std::endl;
	    
	     if (non_phys) {
	       flag=true;
		 
	     }
        }
    }
   delete[] hij;
   delete[] l;
   return flag;
}
int main() {
  int N, xsize;
  double ess, xss, xlower, xupper, elower, eupper;

  std::cout << "Enter the size of the Hankel matrix: ";
    std::cin >> N;
 
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

    const double& xinter = xupper - xlower;
    const double& einter = eupper - elower;
    const int xs2 = 5000000; /// max xsize, fixed before compiling to avoid dynamic alloc.
    double* ees = new double[xs2];
    double* xxs = new double[xs2];
    // std::fill(ees, ees + xs2, 0);
    // std::fill(xxs, xxs + xs2, 0);
    const auto startTime = high_resolution_clock::now();
      
    bool flag=false;
      
#pragma omp parallel for private(xss, ess, flag) shared(ees, xxs, N, xlower, xinter, elower, einter, xsize, g, w)
    for (int i = 0; i < xsize; ++i) {
        xss = xlower + i * xinter / xsize;
        for (int j = 0; j < xsize; ++j) {
	    ess = elower + j * einter / xsize;
	    flag=getCholesky(ess,xss, N);
	    //std::cout<<"flag "<<flag<<std::endl;
            if(!flag) {
	      ees[i*xsize+j]=ess;
	      xxs[i*xsize+j]=xss;  
	    //     // std::cout<<"ess: "<<ess<<std::endl;
	    //     // std::cout<<"ees array: "<<ees[i+j]<<std::endl;
            }  
	}
    }
           
    std::ofstream xsfile("xs.txt");
    std::ofstream esfile("es.txt");
    std::ofstream gridfile("grid.txt");
   
    gridfile << N << " " << xsize << std::endl;
   
    for (int i=0; i < xsize; ++i){
        for (int j = 0; j < xsize; ++j) {
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
    return 0;
}
