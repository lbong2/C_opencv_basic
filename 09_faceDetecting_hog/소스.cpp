#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;
void Grad(Mat in, int r, int c, int* X, int* Y)  {
	for (int j = 0; j < r; j++) { // i = x, j = y
		for (int i = 0; i < c; i++) {
			if (i == 0) {
				if(j == 0) X[j * c + i] = in.at<uchar>(j, i + 1) + in.at<uchar>(j + 1, i + 1);
				else if(j == r - 1) X[j * c + i] = in.at<uchar>(j, i + 1) + in.at<uchar>(j - 1, i + 1);
				else X[j * c + i] = in.at<uchar>(j - 1, i + 1) + in.at<uchar>(j, i + 1) + in.at<uchar>(j + 1, i + 1);
			}
			else if (i == c - 1)
			{
				if (j == 0) X[j * c + i] = -in.at<uchar>(j, i - 1) - in.at<uchar>(j + 1, i - 1);
				else if (j == r - 1) X[j * c + i] = -in.at<uchar>(j, i - 1) - in.at<uchar>(j - 1, i - 1);
				else X[j * c + i] = -in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j, i - 1) - in.at<uchar>(j + 1, i - 1);
			}
			else if (j == 0) {
				X[j * c + i] = in.at<uchar>(j, i + 1) + in.at<uchar>(j + 1, i + 1) - in.at<uchar>(j, i - 1) - in.at<uchar>(j + 1, i - 1);
			}
			else if (j == r - 1) {
				X[j * c + i] = in.at<uchar>(j - 1, i + 1) + in.at<uchar>(j, i + 1) - in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j, i - 1);
			}
			else
				X[j * c + i] = in.at<uchar>(j - 1, i + 1) + in.at<uchar>(j, i + 1) + in.at<uchar>(j + 1, i + 1)
				- in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j, i - 1) - in.at<uchar>(j + 1, i - 1);
		}
	}
	for (int j = 0; j < r; j++) {
		for (int i = 0; i < c; i++) {
			if (j == 0) {
				if (i == 0) Y[j * c + i] = in.at<uchar>(j + 1, i) + in.at<uchar>(j + 1, i + 1);
				else if (i == c - 1) Y[j * c + i] = in.at<uchar>(j + 1, i) + in.at<uchar>(j + 1, i - 1);
				else Y[j * c + i] = in.at<uchar>(j + 1, i - 1) + in.at<uchar>(j + 1, i) + in.at<uchar>(j + 1, i + 1);

			}
			else if (j == r - 1)
			{
				if (i == 0) Y[j * c + i] = -in.at<uchar>(j - 1, i) - in.at<uchar>(j - 1, i + 1);
				else if (i == c - 1) Y[j * c + i] = -in.at<uchar>(j - 1, i) - in.at<uchar>(j - 1, i - 1);
				else Y[j * c + i] = -in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j - 1, i) - in.at<uchar>(j - 1, i + 1);
			}
			else if (i == 0)
				Y[j * c + i] = in.at<uchar>(j + 1, i) + in.at<uchar>(j + 1, i + 1) - in.at<uchar>(j - 1, i) - in.at<uchar>(j - 1, i + 1);

			else if (i == c - 1)
				Y[j * c + i] = in.at<uchar>(j + 1, i - 1) + in.at<uchar>(j + 1, i) - in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j - 1, i);
			else
				Y[j * c + i] = in.at<uchar>(j + 1, i - 1) + in.at<uchar>(j + 1, i) + in.at<uchar>(j + 1, i + 1)
				- in.at<uchar>(j - 1, i - 1) - in.at<uchar>(j - 1, i) - in.at<uchar>(j - 1, i + 1);
		}
	}
}
void make_histo(Mat in, int r, int c, int* X, int* Y, int* histo) {
	int mag;
	int x_val, y_val;
	double idx;
	int histo_height = 0;
	for (int i = 0; i <= c - 16; i += 8) {
		for (int j = 0; j <= r - 16; j += 8) {
			
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++) {
					x_val = X[(j + y) * c + i + x];
					y_val = Y[(j + y) * c + i + x];
					mag = sqrt(x_val * x_val + y_val * y_val);
					idx = (atan2(y_val, x_val) >= 0) ? (atan2(y_val, x_val) * 57.29577951) : ((atan2(y_val, x_val) * 57.29577951) + 180);
					histo[histo_height * 9 + (int)(idx / 20.)] += mag;
				}
			}
			histo_height++;
			
		}
	}

}
void main() {
	Mat ref = imread("face_ref.bmp", IMREAD_GRAYSCALE);
	Mat tar = imread("face_tar.bmp", IMREAD_GRAYSCALE);
	Mat result = imread("face_tar.bmp", IMREAD_GRAYSCALE);
	Mat sim = Mat::zeros(tar.rows, tar.cols, CV_8UC1);
	int* ref_X, * ref_Y, *ref_histo;
	int* tar_X, * tar_Y, * tar_histo;
	ref_X = (int*)calloc(ref.cols * ref.rows, sizeof(int));
	ref_Y = (int*)calloc(ref.cols * ref.rows, sizeof(int));
	ref_histo = (int*)calloc(9 * (ref.cols / 8 - 1) * (ref.rows / 8 - 1), sizeof(int));

	tar_X = (int*)calloc(tar.cols * tar.rows, sizeof(int));
	tar_Y = (int*)calloc(tar.cols * tar.rows, sizeof(int));

	int max = 0, min = INT_MAX;

	Grad(ref, ref.rows, ref.cols, ref_X, ref_Y);
	make_histo(ref, ref.rows, ref.cols, ref_X, ref_Y, ref_histo);
	Grad(tar, tar.rows, tar.cols, tar_X, tar_Y);
	
	int mag;
	int x_val, y_val;
	float idx = 0.;
	int histo_height;
	for (int b = 0; b <= tar.rows - 32; b++) { // y방향으로 이동하게 만듦 
		for (int a = 0; a <= tar.cols - 32; a++) { // x방향으로 이동하게 만듦

			histo_height = 0;
			tar_histo = (int*)calloc(9 * (ref.cols / 8 - 1) * (ref.rows / 8 - 1), sizeof(int));
			for (int i = 0; i <= ref.cols - 16; i += 8) {
				for (int j = 0; j <= ref.rows - 16; j += 8) {
					
					for (int x = 0; x < 16; x++) {
						for (int y = 0; y < 16; y++) {
							x_val = tar_X[(j + y + b) * tar.cols + i + x + a];
							y_val = tar_Y[(j + y + b) * tar.cols + i + x + a];
							mag = sqrt(x_val * x_val + y_val * y_val);
							idx = (atan2(y_val, x_val) >= 0) ? (atan2(y_val, x_val) * 57.29577951) : ((atan2(y_val, x_val) * 57.29577951) + 180);
							tar_histo[histo_height * 9 + (int)(idx / 20.)] += mag;
						}
					}
					histo_height++;
				}
			}
			
			// HOG 끼리 뺄셈해서 각각의 합 출력 
			int sum = 0;
			for(int p = 0; p < 81; p++)
				sum += (tar_histo[p] - ref_histo[p]) >= 0 ? 
						(tar_histo[p] - ref_histo[p]) : -(tar_histo[p] - ref_histo[p]);
			if (sum > max)
				max = sum;
			if (sum < min)
				min = sum;
			if (sum < 130000)
				rectangle(result, Rect(Point(a, b), Point(a + 32, b + 32)), Scalar(0, 0, 255), 3, 8, 0);
			//printf("%d\n", sum);

			free(tar_histo);
		}
	}
	printf("max : %d min : %d\n", max, min);
	imshow("result", result);
	waitKey(0);

	free(ref_X);
	free(ref_Y);
	free(tar_X);
	free(tar_Y);
	free(ref_histo);
	


}