#include<cstdio>
#include<cmath>
#include<cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;

void main()
{
	VideoCapture capture(0);
	Mat frame;
	if (!capture.isOpened()) {
		printf("Couldn¡¯t open the web camera¡¦\n");
		return;
	}
	while (true) {
		capture >> frame;
		
		imshow("Video", frame);
		if (waitKey(30) >= 0) break;
	}
}
