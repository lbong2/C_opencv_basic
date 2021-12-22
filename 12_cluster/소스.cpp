#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



#include <opencv2/ximgproc.hpp>


#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace cv::ximgproc;
using namespace std;

int main() {

	Mat image = imread("group.jpg", CV_8UC3);
	createSuperpixelLSC(image, 10, 0.075f);
	return 0;
}