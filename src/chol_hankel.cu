
/* To compile: nvcc -o main main.cu -O3 -std=c++17 -lcublas -lcusolver -Xcompiler -fopenmp
*/
#include "cuda_runtime.h"
//#include "device_launch_paraMeters.h"

#include<iostream>
#include <fstream>
#include<iomanip>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include <omp.h>

#include "cuda_runtime.h"
//#include "device_launch_paraMeters.h"
#include "include/cx.h"
#include <thrust/device_free.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>


#define prec_save 10

/******************************************/
/* SET Hankel MATRIX */
/******************************************/

const double g = 1;
const double w = -5;

__host__ double hxs(int s, double E, double x2) {
    if (s % 2 == 1 || s < 0) {
        return 0;
    } else if (s == 0) {
        return 1;
    } else if (s == 2) {
        return x2;
    } else {
        return (4 * (s - 3) * E * hxs(s - 4, E, x2) + (s - 3) * (s - 4) * (s - 5) * hxs(s - 6, E, x2) - 4 * w * (s - 2) * hxs(s - 2, E, x2)) / (4 * g * (s - 1));
    }
}

__device__ void dxs(double * __restrict d_x, int s, double E, double x2) {
    d_x[0]=1;
    d_x[2]=x2;
    if (s % 2 == 1 || s < 0) {
         d_x[s]=0;
    }
    else if (s > 2) {
        d_x[s]= (4 * (s - 3) * E * d_x[s - 4] + (s - 3) * (s - 4) * (s - 5) * d_x[s - 6] - 4 * w * (s - 2) * d_x[s - 2]) / (4 * g * (s - 1));
    }
}


__host__ void setHankel(double * __restrict h_h, double e, double x2, const int N) {

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      h_h[i * N + j] = hxs(i + j, e, x2);
        }
    }
}

__global__ void gpusetHankel(double * __restrict d_h, double * __restrict d_x, double e, double x2, const int N) {

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
     int  s = i+j;
	dxs(d_x, s, e, x2);
	d_h[i * N + j] =d_x[s];
        }
    }
}

/************************************/
/* SAVE REAL ARRAY FROM CPU TO FILE */
/************************************/
template <class T>
void saveCPUrealtxt(const T * h_in, const char *filename, const int M) {

    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
    outfile.close();

}

/************************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
void saveGPUrealtxt(const T * d_in, const char *filename, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));

    //    gpuErrchk(cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost));
     cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost);

    std::ofstream outfile;
    outfile.open(filename);
    for (int i = 0; i < M; i++) outfile << std::setprecision(prec_save) << h_in[i] << "\n";
    outfile.close();

}


/************************************/
/* SAVE REAL ARRAY FROM GPU TO FILE */
/************************************/
template <class T>
bool checknan(const T * d_in, const int M) {

    T *h_in = (T *)malloc(M * sizeof(T));
    bool flag = false;
     cudaMemcpy(h_in, d_in, M * sizeof(T), cudaMemcpyDeviceToHost);

     for (int i = 0; i < M; i++){
       flag = isnan(h_in[i]);
       if (flag){
	 return flag;
	 break;
       }
  }
     return flag;
}

/********/
/* MAIN */
/********/
int main(){
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
  // const int xs2 = 5000000; /// max xsize, fixed before compiling to avoid dynamic alloc.
  double *ees = (double *)malloc(xsize * xsize * sizeof(double));
  double *xxs = (double *)malloc(xsize * xsize * sizeof(double));
  
      /***********************/
      /* SETTING THE PROBLEM */
      /***********************/  
  // --- CUDA solver initialization
  cusolverDnHandle_t solver_handle;
  cusolverDnCreate(&solver_handle);
  // --- CUBLAS initialization
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
    
   
      #pragma omp parallel for private(xss, ess) shared(ees, xxs, N, xlower, xinter, elower, einter, xsize, g, w)
  for (int i = 0; i < xsize; ++i) {
    xss = xlower + i * xinter / xsize;
    for (int j = 0; j < xsize; ++j) {
      ess = elower + j * einter / xsize;
   
      // --- Allocate device space for the input matrix 
     
      // thrust::device_vector<double> dh(N * N * sizeof(double) ); // GPU hankel buffer 
      // double *dhptr = thrust::raw_pointer_cast(&dh[0]); // get hankel pointer
    thrust::device_ptr<double> device_h_ptr = thrust::device_malloc<double>(N * N * sizeof(double));
    double * dhptr = thrust::raw_pointer_cast(device_h_ptr);
       // --- Allocate device space for xs 
      //  double *d_h; cudaMalloc(&d_h, N * N * sizeof(double));
      thrust::device_vector<double> dx(N * sizeof(double) ); // GPU xs buffer 
      double *dxptr = thrust::raw_pointer_cast(&dx[0]); // get dx pointer
   
      
         
      /****************************************/
      /* COMPUTING THE CHOLESKY DECOMPOSITION */
      /****************************************/
      // --- cuSOLVE input/output parameters/arrays
      int work_size = 0;
      int *devInfo;     cudaMalloc(&devInfo, sizeof(int));

      int threads = 256;
      int blocks = (N+threads-1)/threads;  // ensure threads*blocks ≥ steps

      gpusetHankel<<<blocks,threads>>>(dhptr, dxptr,  ess, xss, N);

      // --- CUDA CHOLESKY initialization
      cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_LOWER, N, dhptr, N, &work_size);
    // --- CUDA POTRF execution
      double *work;   cudaMalloc(&work, work_size * sizeof(double));
      cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_LOWER, N, dhptr, N, work, work_size, devInfo);    
      // int devInfo_h = 0;  cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
     bool flag = checknan(dhptr, N * N);
     //	 std::cout << "flag :" << flag << "\n";
        if (!flag){
	  ees[i*xsize+j]=ess;
	  xxs[i*xsize+j]=xss;  
      }
      // --- At this point, the lower triangular part of A contains the elements of L. 
      /***************************************/
      /* CHECKING THE CHOLESKY DECOMPOSITION */
      /***************************************/
      
      //saveCPUrealtxt(h_h, "solveSquareLinearSystemCholeskyCUDA\\h_A.txt", N * N);
      //saveGPUrealtxt(d_h, "solveSquareLinearSystemCholeskyCUDA\\d_A.txt", N * N);
      
	 // free memory
       	//cudaFree(dh);
    thrust::device_free(device_h_ptr);
    }
  }
   cusolverDnDestroy(solver_handle);              
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

//  cusolveSafeCall(cusolverDnDestroy(solver_handle));
    return 0;

}
