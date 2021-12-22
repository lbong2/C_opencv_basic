#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<stdio.h>
#include<cmath>
#pragma warning (disable : 4996)
using namespace cv;

int limit(double a) {
    if (a > 255.)
        return 255;
    else return a;
}
void myResize(Mat src, Mat dst, int cx, int cy) {
    int i, j, x1, y1, x2, y2;
    double rx, ry, p, q;
    double B_val, G_val, R_val;

    for (j = 0; j < cy; j++)
        for (i = 0; i < cx; i++)
        {
            rx = (src.cols - 1) * i / (cx - 1.); // factor에 의한 src픽셀 내 어떠한것 예를들어 factor = 1.5 rx = 0.6666.... 
            ry = (src.rows - 1) * j / (cy - 1.);

            x1 = rx; // rx보다 작은 최대의 정수 
            y1 = ry;

            x2 = x1 + 1; // rx보다 큰 최소의 정수 
            if (x2 == src.cols) x2 = src.cols - 1;
            y2 = y1 + 1; 
            if (y2 == src.rows) y2 = src.rows - 1;

            p = rx - (double)x1; // rx - x1이니까 
            q = ry - (double)y1;

            B_val = (1. - p) * (1. - q) * src.at<Vec3b>(y1, x1)[0] 
                + p * (1. - q) * src.at<Vec3b>(y1, x2)[0]
                + (1. - p) * q * src.at<Vec3b>(y2, x1)[0]
                + p * q * src.at<Vec3b>(y2, x2)[0];
       
            G_val = (1. - p) * (1. - q) * src.at<Vec3b>(y1, x1)[1]
                + p * (1. - q) * src.at<Vec3b>(y1, x2)[1]
                + (1. - p) * q * src.at<Vec3b>(y2, x1)[1]
                + p * q * src.at<Vec3b>(y2, x2)[1];

            R_val = (1. - p) * (1. - q) * src.at<Vec3b>(y1, x1)[2]
                + p * (1. - q) * src.at<Vec3b>(y1, x2)[2]
                + (1. - p) * q * src.at<Vec3b>(y2, x1)[2]
                + p * q * src.at<Vec3b>(y2, x2)[2];
            
            dst.at<Vec3b>(j, i)[0] = (limit(B_val + .5));
            dst.at<Vec3b>(j, i)[1] = (limit(G_val + .5));
            dst.at<Vec3b>(j, i)[2] = (limit(R_val + .5));
            
        }
}
void main() {
	Mat ipImg = imread("test.png", IMREAD_COLOR);
	float f;
	printf("Input factor value >> ");
	scanf("%f", &f);
	int cx = round(ipImg.cols * f);
	int cy = round(ipImg.rows * f);
	Mat result = Mat(cy, cx, CV_8UC3);


	//resize(ipImg, result, Size(cx, cy),0 ,0 , 1);
    myResize(ipImg, result, cx, cy);
    
	imshow("original", ipImg);
	imshow("result", result);
	waitKey(5000);
}