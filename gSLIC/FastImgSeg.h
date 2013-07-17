#ifndef __CUDA_SUPERPIXELSEG__
#define __CUDA_SUPERPIXELSEG__

#include "cudaUtil.h"
#include "cudaSegSLIC.h"

class FastImgSeg
{

public:
	unsigned char* sourceImage;
	unsigned char* markedImg;
	int* segMask;
	SLICClusterCenter* centerList;
	int nMaxSegs;

private:

	int width;
	int height;
	int nSeg;

	bool bSegmented;
	bool bImgLoaded;
	bool bMaskGot;

public:
	FastImgSeg();
	FastImgSeg(int width,int height,int dim,int nSegments);
	~FastImgSeg();

	void initializeFastSeg(int width,int height,int nSegments);
	void clearFastSeg();
	void changeClusterNum(int nSegments);

	void LoadImg(unsigned char* imgP);
	void DoSegmentation(SEGMETHOD eMethod, double weight);
	void Tool_GetMarkedImg();
	void Tool_GetFilledImg();
	void Tool_DrawSites();
};

#endif
