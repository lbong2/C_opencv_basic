#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<stdio.h>
#define BLK 60
using namespace cv;
void main() {

	Mat ipImg = imread("Three_Five.jpg", IMREAD_COLOR);
	int cx = ipImg.cols / 2;
	int cy = ipImg.rows / 2;
	int avg;

	for (int i = 0; i < ipImg.cols; i++) {
		for (int j = 0; j < ipImg.rows; j++) {
			avg = (ipImg.at<Vec3b>(j, i)[0] + ipImg.at<Vec3b>(j, i)[1] + ipImg.at<Vec3b>(j, i)[2]) / 3;
			if (avg < 200)
			{
				ipImg.at<Vec3b>(j, i)[0] = 0;
				ipImg.at<Vec3b>(j, i)[1] = 0;
				ipImg.at<Vec3b>(j, i)[2] = 0;
			}
			else {
				ipImg.at<Vec3b>(j, i)[0] = avg;
				ipImg.at<Vec3b>(j, i)[1] = avg;
				ipImg.at<Vec3b>(j, i)[2] = avg;
			}
		}
	}
	imshow("Result", ipImg);
	imwrite("G_Th_F.jpg", ipImg);
	waitKey(5000);
}
