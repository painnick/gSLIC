#include "FastImgSeg.h"
#include "cudaSegEngine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace std;

FastImgSeg::FastImgSeg(int w,int h, int d, int nSegments)
{
	initializeFastSeg(w, d, nSegments);
}

FastImgSeg::FastImgSeg()
{
	// Do nothing
}

FastImgSeg::~FastImgSeg()
{
	clearFastSeg();
}


void FastImgSeg::changeClusterNum(int nSegments)
{
	nSeg=nSegments;
}

void FastImgSeg::initializeFastSeg(int w,int h, int nSegments)
{
	int nAvailSegs=(iDivUp(w,BLCK_SIZE))*(iDivUp(h,BLCK_SIZE));
	if(nAvailSegs < nSegments) {
		printf("Max no. of segment is %d\n", nAvailSegs);
		nSegments = nAvailSegs;
	}

	// MaxSegs should be same on InitCUDA()@CudaSegEngine.cu
	int nMaxSegs=(iDivUp(w,BLCK_SIZE)*2)*(iDivUp(h,BLCK_SIZE)*2);

	width=w;
	height=h;
	nSeg=nSegments;

	segMask=(int*) malloc(width*height*sizeof(int));
	markedImg=(unsigned char*)malloc(width*height*4*sizeof(unsigned char));

	centerList=(SLICClusterCenter*)malloc(nMaxSegs*sizeof(SLICClusterCenter));

	InitCUDA(width,height,nSegments,SLIC);

	bImgLoaded=false;
	bSegmented=false;
}

void FastImgSeg::clearFastSeg()
{
	free(segMask);
	free(markedImg);
	free(centerList);
	TerminateCUDA();
	bImgLoaded=false;
	bSegmented=false;
}


void FastImgSeg::LoadImg(unsigned char* imgP)
{
	sourceImage=imgP;
	CUDALoadImg(sourceImage);
	bSegmented=false;
}

void FastImgSeg::DoSegmentation(SEGMETHOD eMethod, double weight)
{
	clock_t start,finish;

	start=clock();
	CudaSegmentation(eMethod, weight);
	finish=clock();
	printf("clustering:%f ",(double)(finish-start)/CLOCKS_PER_SEC);

	CopyMaskDeviceToHost(segMask);
	CopyCenterListDeviceToHost(centerList);

	start=clock();
	int nMaxSegs=(iDivUp(width,BLCK_SIZE) + 1)*(iDivUp(height,BLCK_SIZE) + 1);
	enforceConnectivity(segMask,width,height,nMaxSegs);
	finish=clock();
	printf("connectivity:%f\n",(double)(finish-start)/CLOCKS_PER_SEC);

	bSegmented=true;
}

void FastImgSeg::Tool_GetMarkedImg()
{
	if (!bSegmented)
	{
		return;
	}

	memcpy(markedImg,sourceImage,width*height*4*sizeof(unsigned char));

	for (int i=1;i<height-1;i++)
	{
		for (int j=1;j<width-1;j++)
		{
			int mskIndex=i*width+j;
			if (segMask[mskIndex]!=segMask[mskIndex+1] 
			|| segMask[mskIndex]!=segMask[(i-1)*width+j]
			|| segMask[mskIndex]!=segMask[mskIndex-1]
			|| segMask[mskIndex]!=segMask[(i+1)*width+j])
			{
				markedImg[mskIndex*4]=0;
				markedImg[mskIndex*4+1]=0;
				markedImg[mskIndex*4+2]=255;
			}
		}
	}

	// DRAW Center
	int nMaxSegs=iDivUp(width,BLCK_SIZE)*iDivUp(height,BLCK_SIZE);
	for(int i = 0; i < nMaxSegs; i ++)
	{
		float2 srcXY = centerList[i].xy;
		int srcX = srcXY.x;
		int srcY = srcXY.y;

		int srcIndex = srcY*width+srcX;

		markedImg[srcIndex*4 + 0] = 255;
		markedImg[srcIndex*4 + 1] = 255;
		markedImg[srcIndex*4 + 2] = 255;
	}
}

void FastImgSeg::Tool_GetFilledImg()
{
	if (!bSegmented)
		return;

	memcpy(markedImg,sourceImage,width*height*4*sizeof(unsigned char));

	int nMaxSegs=(iDivUp(width,BLCK_SIZE)+1)*(iDivUp(height,BLCK_SIZE)+1);

	for (int i=0;i<height;i++)
	{
		for (int j=0;j<width;j++)
		{
			int mskIndex=i*width+j;
			int centerIndex = segMask[mskIndex];

			if(centerIndex >= nMaxSegs) {
				printf("[%s:%d] centerIndex(%d) is greater than or equals nMaxSegs(%d)\n", __FILE__, __LINE__, centerIndex, nMaxSegs);
				continue;
			}

			//============================================================
			// Method 1
			//============================================================
			//float2 srcXY = centerList[centerIndex].xy;
			//int srcX = srcXY.x;
			//int srcY = srcXY.y;

			//int srcIndex = srcY*width+srcX;

			//markedImg[mskIndex*4+0] = sourceImage[srcIndex*4+0];
			//markedImg[mskIndex*4+1] = sourceImage[srcIndex*4+1];
			//markedImg[mskIndex*4+2] = sourceImage[srcIndex*4+2];

			//============================================================
			// Method 2
			//============================================================
			float4 lab = centerList[centerIndex].lab;
			float x = lab.x * 255;
			float y = lab.y * 255;
			float z = lab.z * 255;

			int r = 3.24071 * x + (-1.53726) * y + (-0.498571) * z;
			int g =(-0.969258) * x + 1.87599 * y + 0.0415557 * z;
			int b = 0.0556352 * x + (-0.203996) * y + 1.05707 * z;

			markedImg[mskIndex*4 + 0] = b;
			markedImg[mskIndex*4 + 1] = g;
			markedImg[mskIndex*4 + 2] = r;
		}
	}

	for(int i = 0; i < nMaxSegs; i ++)
	{
		centerList[i].x1 = width;
		centerList[i].y1 = height;
		centerList[i].x2 = -1;
		centerList[i].y2 = -1;
	}

	for (int i=0;i<height;i++)
	{
		for (int j=0;j<width;j++)
		{
			int mskIndex=i*width+j;
			int centerIndex = segMask[mskIndex];

			if(centerIndex < 0 || centerIndex >= nMaxSegs)
				printf("[%s:%d] centerIndex(%d) of (x:%d, y:%d) is not between 0 and nMaxSegs(%d)\n", __FILE__, __LINE__, j, i, centerIndex, nMaxSegs);

			SLICClusterCenter* center = &(centerList[centerIndex]);

			if(j < center->x1)
				center->x1 = j;
			if(j > center->x2)
				center->x2 = j;
			if(i < center->y1)
				center->y1 = i;
			if(i > center->y2)
				center->y2 = i;
		}
	}

	SLICClusterCenter center = centerList[nMaxSegs - 2];

	int x,y;
	for(y = center.y1, x = center.x1; x <= center.x2; x ++)
	{
		int idx = y*width+x;

		markedImg[idx*4 + 0] = 255;
		markedImg[idx*4 + 1] = 0;
		markedImg[idx*4 + 2] = 0;
	}
	for(y = center.y2, x = center.x1; x <= center.x2; x ++)
	{
		int idx = y*width+x;

		markedImg[idx*4 + 0] = 255;
		markedImg[idx*4 + 1] = 0;
		markedImg[idx*4 + 2] = 0;
	}
	for(x = center.x1, y = center.y1; y <= center.y2; y ++)
	{
		int idx = y*width+x;

		markedImg[idx*4 + 0] = 255;
		markedImg[idx*4 + 1] = 0;
		markedImg[idx*4 + 2] = 0;
	}
	for(x = center.x2, y = center.y1; y <= center.y2; y ++)
	{
		int idx = y*width+x;

		markedImg[idx*4 + 0] = 255;
		markedImg[idx*4 + 1] = 0;
		markedImg[idx*4 + 2] = 0;
	}

	// DRAW Center
	for(int i = 0; i < nMaxSegs; i ++)
	{
		float2 srcXY = centerList[i].xy;
		int srcX = srcXY.x;
		int srcY = srcXY.y;

		int srcIndex = srcY*width+srcX;

		markedImg[srcIndex*4 + 0] = 0;
		markedImg[srcIndex*4 + 1] = 0;
		markedImg[srcIndex*4 + 2] = 255;
	}
}