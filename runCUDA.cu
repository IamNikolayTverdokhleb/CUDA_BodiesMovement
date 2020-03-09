#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <cstring>
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <vector>

using namespace std;

const float eps = 10e-10;   /*accuracy*/
const float G = 6.67e-11;   /*gravity const*/
const float tau = 0.001;    /*time step*/
const float t0 = 0;         /*start time*/
const float T = 0.003;      /*total time*/
const int N = 1024 * 16;    /*number of bodies*/
const int numthreads = 128; /*cuda - threads*/
const int blocksize = N / numthreads ; /*size of block*/
const float tau2 = tau / 2.; /*half time step*/


/*
* generate bodies, speed and initial radius
* @param{float*} m - mass
* @param{float*} r - radius
* @param{float*} m - speed
*/
__host__ void body_gen(float* m, float* r, float* v){
	float mas = 0.0;
	float TEMP = 0.0;
//	srand(time(0));
	for (int i = 0; i < N; ++i){
		mas = rand() % 1000000000 + 1000000000;
		m[i] = mas;
			for (int j = 0; j < 3; ++j){
			  TEMP = exp(cos((rand() % 10000)));
			  r[i * 3 + j] = TEMP;
			}
			for (int j = 0; j < 3; ++j){
			  TEMP = exp(cos((rand() % 10000)));
			  v[i * 3 + j] = TEMP;
			}
	}/*end for N*/
}

/* 
* squared
*/
__host__ __device__ inline float pow2(const float x)
{
	return x * x;
}

/*
* generate bodies. if 4 - use inicial condicions, otherwise random
* @param{float*} m - mass
* @param{float*} r - radius
* @param{float*} m - speed
*/
__host__ void ic_input(float* m, float* r, float* v)
{
	if (N != 4) body_gen(m, r, v);
	else
	{
		float mt = 8810324116.227;

		r[0 * 3 + 0] = 1.; r[0 * 3 + 1] = 0.; r[0 * 3 + 2] = 0.;
		v[0 * 3 + 0] = 0.; v[0 * 3 + 1] = 0.9; v[0 * 3 + 2] = 0.;
		m[0] = mt;

		r[1 * 3 + 0] = 0.; r[1 * 3 + 1] = 1.; r[1 * 3 + 2] = 0.;
		v[1 * 3 + 0] = -0.9; v[1 * 3 + 1] = 0.; v[1 * 3 + 2] = 0.;
		m[1] = mt;

		r[2 * 3 + 0] = -1.; r[2 * 3 + 1] = 0.; r[2 * 3 + 2] = 0.;
		v[2 * 3 + 0] = 0.; v[2 * 3 + 1] = -0.9; v[2 * 3 + 2] = 0.;
		m[2] = mt;

		r[3 * 3 + 0] = 0.; r[3 * 3 + 1] = -1.; r[3 * 3 + 2] = 0.;
		v[3 * 3 + 0] = 0.9; v[3 * 3 + 1] = 0.; v[3 * 3 + 2] = 0.;
		m[3] = mt;

	}

}

/*probably not needed*/
__host__ void ic_check(const float* m,const float* r,const float* v)
{
	cout << "Input data check " << endl;
	cout << "tau = " << tau << endl;
	cout << "T = " << T << endl;
	cout << "t0 = " << t0 << endl;
	cout << "N = " << N << endl;
	for (int i = 0; i < N; ++i)
	{
		cout << "m = " << m[i] << endl;

		for (int j = 0; j < 3; ++j)
			cout << "r[" << i << ", " << j << "] = " << r[i * 3 + j] << endl;

		for (int j = 0; j < 3; ++j)
			cout << "v[" << i << ", " << j << "] = " << v[i * 3 + j] << endl;

	}
}

/*
* @param{const float*} m - mass
* @param{const float*} r - radius
* @param{float*} a - speed
*/
__global__ void kernel(const float* m, const float* r, float* a)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x; /*number of thread*/

	float temp = 0.0; // 2

	float r0 = r[id * 3]; // 3 
	float r1 = r[id * 3 + 1]; // 4
	float r2 = r[id * 3 + 2] ; // 5 

	float a0 = 0, a1 = 0, a2 = 0; // 6 7 8

	for (int j = 0; j < N; ++j){

		temp = pow2(r0 - r[j * 3]) + pow2(r1 - r[j * 3 + 1]) + pow2(r2 - r[j * 3 + 2]);
		temp = m[j] / max(temp * sqrtf(temp), eps);

		a0 += (r0 - r[j * 3]) * temp;
		a1 += (r1 - r[j * 3 + 1]) * temp;
		a2 += (r2 - r[j * 3 + 2]) * temp;
	}

	a[id * 3] = -a0 * G;
	a[id * 3 + 1] = -a1 * G;
	a[id * 3 + 2] = -a2 * G;

	__syncthreads();
}

__host__ void RK(const float* m, float* r, float* v, const float tau)
{
   	dim3 threads(numthreads);
   	dim3 blocks(blocksize);

	float t = t0;

	float* k1 = new float[3 * N]; // host
	float* k2 = new float[3 * N]; // host
	
	float* r_t = new float[3 * N]; // host
	float* v_t = new float[3 * N]; // host

	float* cu_r = new float[3 * N]; // device
	float* cu_a = new float[3 * N]; // device
	float* cu_m = new float[N];     // device

	cudaMalloc((void**)& cu_a, N * 3 * sizeof(float)); /*allocate memory for acceleretions*/
	cudaMalloc((void**)& cu_r, N * 3 * sizeof(float)); /*allocate memory for radiuses*/
	cudaMalloc((void**)& cu_m, N * sizeof(float));     /*allocate memory for masses*/

	cudaMemcpy(cu_m, m, N * sizeof(float), cudaMemcpyHostToDevice); /*copy masses from host to device*/

	do{
		t += tau;

		cudaMemcpy(cu_r, r, 3 * N * sizeof(float), cudaMemcpyHostToDevice);  /*copy radiuses from host to device*/
		kernel << < blocks, threads >> > (cu_m, cu_r, cu_a); 				 /*run kernel*/
		cudaMemcpy(k1, cu_a, 3 * N * sizeof(float), cudaMemcpyDeviceToHost); /*copy accelerations to k1 from device to host*/

		for (int i = 0; i < N; ++i){ /*first rk step*/
			for (int j = 0; j < 3; ++j){
				float vij = v[i * 3 + j];
				r_t[i * 3 + j] = r[i * 3 + j] + vij  * tau;
				v_t[i * 3 + j] = vij + k1[i * 3 + j] * tau;
			}
		}

		cudaMemcpy(cu_r, r_t, 3 * N * sizeof(float), cudaMemcpyHostToDevice); /*copy radiuses_t from host to device*/
		kernel << < blocks, threads >> > (cu_m, cu_r, cu_a); 				  /*run kernel*/
		cudaMemcpy(k2, cu_a, 3 * N * sizeof(float), cudaMemcpyDeviceToHost);  /*copy accelerations to k2 from device to host*/

		for (int i = 0; i < N; ++i){ /*second rk step*/
			for (int j = 0; j < 3; ++j)
				r[i * 3 + j] += (v[i * 3 + j] + v_t[i * 3 + j]) * tau2;
				v[i * 3 + j] += (k1[i * 3 + j] + k2[i * 3 + j]) * tau2;
		}

		if (N == 4) /*calculate difference between our and reference solutions in point t = 10*/
			if (fabs(t - 10) < 1e-10){
				cout << "error = " << \
					sqrt(pow2(r[3 * 2] - 2.5723673494326125) + pow2(r[3 * 2 + 1] - 4.2292866972437615e-7) + pow2(r[3 * 2 + 2])) << endl;
			}


	} while (t <= T);

	delete[] k2;
	delete[] k1;
	delete[] r_t;
	delete[] v_t;

	cudaFree(cu_r);
	cudaFree(cu_a);
	cudaFree(cu_m);
}

int main()
{

	float* m = new float[N]; 
	float* r = new float[3 * N];
	float* v = new float[3 * N];

	ic_input(m, r, v);

	cudaError_t SD; /*if unable to connect to videocart*/

	SD = cudaSetDevice(0);
	if (SD != cudaSuccess)
	{
		cout << "Could not connect to cuda device." << endl;
		return (-1);
	}

	cudaEvent_t start, finish; /*events to count time*/

	cudaEventCreate(&start); cudaEventCreate(&finish); /*create events*/

	cudaEventRecord(start); cudaEventSynchronize(start); /*start time*/

	RK(m, r, v, tau); /*calculations*/

	cudaEventRecord(finish); cudaEventSynchronize(finish); /*finish time*/

	float dt; /*finish time - start time*/

	cudaEventElapsedTime(&dt, start, finish); /*count dt*/

	cout << "number of bodyes = " << N << endl;
	cout << "blocksize = " << blocksize << endl;
	cout << "numthreads = " << numthreads << endl; 
	cout << "time = " << dt / 1000.0  / ( (T - t0) / tau - 1)<< endl; /*one step time*/

	cudaEventDestroy(start);
	cudaEventDestroy(finish);

	delete[] m;
	delete[] r;
	delete[] v;

	return 0;
}