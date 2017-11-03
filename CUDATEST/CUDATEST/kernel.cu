#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//������㺯��
//X�������
//R���ο���
//Return���������ο���֮��ľ���
__device__ float CalDistance(float X, float R)
{
	float result = 0;
	result = sqrt((X - R)*(X - R));
	return result;
}

//��0-10�ֲ����㺯��
//i����i���㣨ע���0��ʼ��
//N������Ŀ
//Return����i�����ֵ
__device__ float CalScale(int i, int N)
{
	float result = 0;
	result = i*(10 - 0) / (N - 1.0);
	return result;
}

//�˺���
//dev_D:�������
//N: ����
//R: �ο���
__global__ void DistKernel(float *dev_D, int N, float R)
{
	int i = threadIdx.x;
	float result;
	float scale;
	scale = CalScale(i, N);
	result = CalDistance(scale, R);
	dev_D[i] = result;
}

int main()
{
	int const N = 100;//����
	float R = 6.0f;//�ο���
	float D[N] = { 0 };//������
	float* dev_D = 0;

	//GPU״̬���
	cudaError_t cudaStatus = cudaSetDevice(0);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return  -1;
	}

	cudaStatus = cudaMalloc((void**)&dev_D, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_D);
		return  -1;
	}

	//���ú˺���
	DistKernel << <1, N >> >(dev_D, N, R);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_D);
		return  -1;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		cudaFree(dev_D);
		return  -1;
	}

	cudaStatus = cudaMemcpy(D, dev_D, N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_D);
		return  -1;
	}

	//�ͷ�GPU��Դ
	cudaFree(dev_D);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -1;
	}

	printf("D[50]=%f", D[50]);
	getchar();
	return 0;
}