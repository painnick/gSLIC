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

CvSubdiv2D* InitSubdivision(CvMemStorage* storage, CvRect rect)
{
	CvSubdiv2D* subdiv;

	subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv), sizeof(CvSubdiv2DPoint), sizeof(CvQuadEdge2D), storage);
	
	cvInitSubdivDelaunay2D(subdiv, rect);

	return subdiv;
}

void LocatePoint(CvSubdiv2D* subdiv, CvPoint2D32f fp)
{
	CvSubdiv2DEdge e;
	CvSubdiv2DEdge e0 = 0;
	CvSubdiv2DPoint* p = 0;

	cvSubdiv2DLocate(subdiv, fp, &e0, &p);

	if(e0) {
		e = e0;
		
		do {
			e = cvSubdiv2DGetEdge(e, CV_NEXT_AROUND_LEFT);
		}
		while( e != e0 );
	}
}

CvSeq* ExtractSeq(CvSubdiv2DEdge edge, CvMemStorage* storage)
{
	CvSeq* pSeq = NULL;

  CvSubdiv2DEdge egTemp = edge;
  int i, nCount = 0;
  CvPoint* buf = 0;

  // count number of edges in facet
  do {
    nCount ++;
    egTemp = cvSubdiv2DGetEdge(egTemp, CV_NEXT_AROUND_LEFT );
  }
  while(egTemp != edge);

  buf = (CvPoint*)malloc(nCount * sizeof(buf[0]));

  // gather points
  egTemp = edge;
    
  for( i = 0; i < nCount; i++ ) {
    CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg(egTemp);
    
    if(!pt)
			break;

		CvPoint ptInsert = cvPoint( cvRound(pt->pt.x), cvRound(pt->pt.y));
		
    buf[i] = ptInsert;
    
    egTemp = cvSubdiv2DGetEdge(egTemp, CV_NEXT_AROUND_LEFT );
  }

  if(i == nCount) {
		pSeq = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour), sizeof(CvPoint), storage);
		
		for( i = 0; i < nCount; i ++)
			cvSeqPush(pSeq, &buf[i]);
  }
  
  free( buf );
  
  return pSeq;
}

void DrawVoronoiDiagram(IplImage* image, SLICClusterCenter* centerList, int listSize) {

	CvRect cvRect = {0, 0, image->width, image->height};

	CvMemStorage* pStorage = cvCreateMemStorage(0);

	// Init Subdivision
	CvSubdiv2D* pSubDiv = InitSubdivision(pStorage, cvRect);

	// Add sites
	for(int i = 0; i < listSize; i ++) {
		SLICClusterCenter center = centerList[i];

		float x = center.xy.x;
		float y = center.xy.y;

		if(0 <= x && x < image->width && 0 <= y && y < image->height) {
			CvPoint2D32f fPoint = cvPoint2D32f(x, y);
			LocatePoint(pSubDiv, fPoint);
			cvSubdivDelaunay2DInsert(pSubDiv, fPoint);
		}
	}

	// Calculate voronoi tessellation
	cvCalcSubdivVoronoi2D(pSubDiv);

	// Draw edges
	int nEdgeCount = pSubDiv->edges->total;
	int nElementSize = pSubDiv->edges->elem_size;

	CvPoint** ppPoints = new CvPoint*[1];
	ppPoints[0] = new CvPoint[2048];
	int pnPointCount[1];

  CvSeqReader reader;
  
  cvStartReadSeq( (CvSeq*)(pSubDiv->edges), &reader, 0);

  for(int i = 0; i < nEdgeCount; i++) {
    CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

    if( CV_IS_SET_ELEM( edge )) {
			CvSubdiv2DEdge e = (CvSubdiv2DEdge)edge;

			CvSeq *pSeq = ExtractSeq(cvSubdiv2DRotateEdge( e, 1 ), pStorage);

			if(pSeq != NULL) {
				pnPointCount[0] = pSeq->total;
	
				for(int j = 0; j < pSeq->total; j ++) {
					CvPoint pt = *CV_GET_SEQ_ELEM(CvPoint, pSeq, j);
					ppPoints[0][j] = cvPoint(pt.x, pt.y);
				}

				cvPolyLine(image, ppPoints, pnPointCount, 1, -1, CV_RGB(0, 0, 255));
			}
    }

    CV_NEXT_SEQ_ELEM( nElementSize, reader);
  }

	delete [] ppPoints[0];
	delete [] ppPoints;

  cvReleaseMemStorage(&pStorage);
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

	//mySeg->centerList
	DrawVoronoiDiagram(frame, mySeg->centerList, mySeg->nMaxSegs);

	cvShowImage("frame",frame);

	cvWaitKey(0);

	cvSaveImage("photo_segmented.png", frame);

	cvDestroyWindow( "frame" );

	cvFree(&frame);
	delete mySeg;

	return 0;
}
