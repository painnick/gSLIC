// Test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// Copy cvImage to memory buffer
void CvImgToBuffer(IplImage* frame, unsigned char* imgBuffer) {
	for (int i=0;i<frame->height;i++) {
		for (int j=0;j<frame->width;j++) {
			int bufIdx=(i*frame->width+j)*4;

			imgBuffer[bufIdx]=CV_IMAGE_ELEM(frame,unsigned char,i,j*3);
			imgBuffer[bufIdx+1]=CV_IMAGE_ELEM(frame,unsigned char,i,j*3+1);
			imgBuffer[bufIdx+2]=CV_IMAGE_ELEM(frame,unsigned char,i,j*3+2);
		}
	}
}

// Copy memory buffer to cvImage
void OutputImgToCvImg(unsigned char* markedImg, IplImage* frame) {
	for (int i=0;i<frame->height;i++) {
		for (int j=0;j<frame->width;j++) {
			int bufIdx=(i*frame->width+j)*4;
			CV_IMAGE_ELEM(frame,unsigned char,i,j*3)=markedImg[bufIdx];
			CV_IMAGE_ELEM(frame,unsigned char,i,j*3+1)=markedImg[bufIdx+1];
			CV_IMAGE_ELEM(frame,unsigned char,i,j*3+2)=markedImg[bufIdx+2];
		}
	}
}

int _tmain(int argc, _TCHAR* argv[]) {
	IplImage* frame = cvLoadImage("photo.jpg");

	FastImgSeg* mySeg = new FastImgSeg();
	mySeg->initializeFastSeg(frame->width,frame->height, 2000);

	unsigned char* imgBuffer=(unsigned char*)malloc(frame->width * frame->height * sizeof(unsigned char) * 4);

	CvImgToBuffer(frame, imgBuffer);

	mySeg->LoadImg(imgBuffer);

	cvNamedWindow("frame",0);

	float weight = 0.3f;
	mySeg->DoSegmentation(XYZ_SLIC, weight);

	mySeg->Tool_GetFilledImg();
	//mySeg->Tool_GetMarkedImg();
	//mySeg->Tool_DrawSites();

	OutputImgToCvImg(mySeg->markedImg, frame);

	cvShowImage("frame",frame);

	cvWaitKey(0);

	cvSaveImage("photo_segmented.png", frame);

	cvDestroyWindow( "frame" );

	cvFree(&frame);
	delete mySeg;

	return 0;
}
