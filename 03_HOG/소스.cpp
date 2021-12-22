#include<cstdio>
#include<cmath>
#include<cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#pragma warning(disable : 4996)
using namespace cv;
#define PI 3.141592
FILE* fp;
void Grad(int r, int c, int** X, int** Y, Mat in) {
	for (int i = 0; i < c; i++) {
		for (int j = 0; j < r; j++) {
			if (i == 0) {
				X[j][i] = (int)in.at<uchar>(j, i + 1);
			}
			else if (i == c - 1)
			{
				X[j][i] = (int)in.at<uchar>(j, i - 1) * -1;
			}
			else 
				X[j][i] = (int)in.at<uchar>(j, i + 1) - (int)in.at<uchar>(j, i - 1);
		}
	}
	for (int i = 0; i < c; i++) {
		for (int j = 0; j < r; j++) {
			if (j == 0) {
				Y[j][i] = (int)in.at<uchar>(j + 1, i) * -1;
			}
			else if (j == r - 1)
			{
				Y[j][i] = (int)in.at<uchar>(j - 1, i);
			}
			else
				Y[j][i] = (int)in.at<uchar>(j - 1, i) - (int)in.at<uchar>(j + 1, i);
		}
	}
}
void make_histo(Mat ipimg, int** X, int** Y, const char* filename) {
	int mag;
	int x_val, y_val;
	double idx;
	fp = fopen(filename, "wt");
	for (int i = 0; i < ipimg.cols - 16; i += 8) {
		for (int j = 0; j < ipimg.rows - 16; j += 8) {
			int histo[9] = { 0, };
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++) {
					x_val = X[j + y][i + x];
					y_val = Y[j + y][i + x];
					mag = x_val * x_val + y_val * y_val;
					idx = (atan2(y_val, x_val) >= 0) ? (atan2(y_val, x_val) * 57.29577951) : ((atan2(y_val, x_val) * 57.29577951) + 180);
					histo[(int)(idx / 20.)] += mag;
				}
			}
			fprintf(fp, "%d %d %d %d %d %d %d %d %d ", histo[0], histo[1], histo[2], histo[3], histo[4], histo[5], histo[6], histo[7], histo[8]);
		}
	}
	fclose(fp);
}


void main() {
	Mat cp1 = imread("compare1.bmp", CV_8UC1);
	Mat cp2 = imread("compare2.bmp", CV_8UC1);
	Mat ipimg = imread("assignment3.bmp", CV_8UC1);
	Mat cp3 = imread("compare3.bmp", CV_8UC1);

	int** X, **Y;

	X = (int**)calloc(cp1.rows, sizeof(int*));
	Y = (int**)calloc(cp1.rows, sizeof(int*));
	for (int i = 0; i < cp1.rows; i++) {
		X[i] = (int*)calloc(cp1.cols, sizeof(int));
		Y[i] = (int*)calloc(cp1.cols, sizeof(int));
	}
	Grad(cp1.rows, cp1.cols, X, Y, cp1);
	make_histo(cp1, X, Y, "compare1.csv");
	
	Grad(cp2.rows, cp2.cols, X, Y, cp2);
	make_histo(cp2, X, Y, "compare2.csv");

	Grad(ipimg.rows, ipimg.cols, X, Y, ipimg);
	make_histo(ipimg, X, Y, "assinment03.csv");

	Grad(cp3.rows, cp3.cols, X, Y, cp3);
	make_histo(cp3, X, Y, "compare3.csv");

	for (int i = 0; i < cp1.rows; i++) {
		free(X[i]);
		free(Y[i]);
	}
	free(X);
	free(Y);
}