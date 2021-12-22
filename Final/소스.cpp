#include<opencv2/imgproc.hpp>
#include<opencv2/photo.hpp>
#include<opencv2/highgui.hpp>
#include<stdlib.h>
#include<iostream>
#include<stdio.h>
#pragma warning(disable: 4996)

#define width 6
#define height 8
using namespace cv;
using namespace std;
float img[48] = { 3, 3, 4, 9, 10, 10, 
3,5,3,4,5,10,2,
3,3,3,5,9,1,5,3,6,6,9
,3,3,7,8,7,9,4,5,6,9,9,9,8
,7,7,8,9,8,7,7,6,8,8,9};
float R_0[8] = { -0.1, -0.1, 0.1, 0, 0.2, 0.1, 0, 0.1 };
float b_0[4] = { 0.4, 0, 0.3, 0.3 };
int original_pos[4] = { 2, 2, 3, 5 }; // 2개 랜드마크 좌표
int landmark_val[2] = { 3, 9 }; // 2개 랜드마크 밝기 값
/// <summary>
/// LBP 구하는 함수
/// </summary>
/// <param name="in">: 입력 이미지 </param>
/// <param name="bi">: 출력 이미지 </param>
/// <param name="uni">: uniform 적용 여부</param>
Mat cal_LBP(Mat in, int uni) {
	int val;
	cvtColor(in, in, COLOR_BGR2GRAY);
	Mat bi = Mat::zeros(in.rows, in.cols, CV_8UC1);
	for (int y = 0; y < in.rows; y++) {
		for (int x = 0; x < in.cols; x++) {
			val = in.at<uchar>(y, x);
			if (x == 0) {
				if (y == 0) {
					bi.at<uchar>(0, 0) += (val < in.at<uchar>(y, x + 1)) ? 32 : 0;
					bi.at<uchar>(0, 0) += (val < in.at<uchar>(y + 1, x + 1)) ? 16 : 0;
					bi.at<uchar>(0, 0) += (val < in.at<uchar>(y + 1, x)) ? 8 : 0;
				}
				else if (y == in.rows - 1) {
					bi.at<uchar>(in.rows - 1, 0) += (val < in.at<uchar>(y - 1, x)) ? 128 : 0;
					bi.at<uchar>(in.rows - 1, 0) += (val < in.at<uchar>(y - 1, x + 1)) ? 64 : 0;
					bi.at<uchar>(in.rows - 1, 0) += (val < in.at<uchar>(y, x + 1)) ? 32 : 0;
				}
				else {
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x)) ? 128 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x + 1)) ? 64 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x + 1)) ? 32 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x + 1)) ? 16 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x)) ? 8 : 0;
				}
			}
			else if (x == in.cols - 1) {
				if (y == 0) {
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x)) ? 8 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x - 1)) ? 4 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x - 1)) ? 2 : 0;
				}
				else if (y == in.rows - 1) {
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x)) ? 128 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x - 1)) ? 2 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x - 1)) ? 1 : 0;
				}
				else {
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x)) ? 128 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x)) ? 8 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x - 1)) ? 4 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x - 1)) ? 2 : 0;
					bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x - 1)) ? 1 : 0;
				}
			}
			else if (y == 0) {
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x + 1)) ? 32 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x + 1)) ? 16 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x)) ? 8 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x - 1)) ? 4 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x - 1)) ? 2 : 0;
			}
			else if (y == in.rows - 1) {
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x)) ? 128 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x + 1)) ? 64 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x + 1)) ? 32 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x - 1)) ? 2 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x - 1)) ? 1 : 0;
			}
			else {
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x)) ? 128 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x + 1)) ? 64 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x + 1)) ? 32 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x + 1)) ? 16 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x)) ? 8 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y + 1, x - 1)) ? 4 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y, x - 1)) ? 2 : 0;
				bi.at<uchar>(y, x) += (val < in.at<uchar>(y - 1, x - 1)) ? 1 : 0;
			}
		}
	}
	if (uni == 1) {
		for (int i = 0; i < in.cols; i++) {
			for (int j = 0; j < in.rows; j++) {
				int temp = bi.at<uchar>(j, i);
				bi.at<uchar>(j, i) = uniform[temp];
			}
		}
	}
	return bi;
}

/// <summary>
/// LBP Histogram 만드는 함수
/// </summary>
/// <param name="en">: 입력 이미지 </param>
/// <param name="bi">: LBP 이미지 </param>
/// <param name="histo">: LBP Histogram 변수 </param>
void makeLBPhisto(Mat bi, float** histo) {
	int count = 0; // histogram 줄 표시용 
	int val;
	for (int y = 0; y <= bi.rows - 32; y += 16) {
		for (int x = 0; x <= bi.cols - 32; x += 16) {

			for (int j = 0; j < 32; j++) {
				for (int i = 0; i < 32; i++) {
					val = (int)bi.at<uchar>(y + j, x + i);
					histo[count][val] += 1. / 256.;
				}
			}
			count++;
		}
	}
}
void update_landmark_SDM(int* pos){
	for (int i = 0; i < width * height; i++) {
		int x1 = original_pos[0] % width;
		int y1 = original_pos[1] / width;
		int x2 = original_pos[3] % width;
		int y2 = original_pos[4] / width;


	}
	
}
void LBP_histogram(int * pos, int * img, int* hist, int * hist1); {
	val = img[width * (y)+x;
	int bi = 0;
	
	for (int i = 0; i < 16; i++) {
		bi += (val < img[width * (y - 1) + x]) ? 128 : 0;
		bi += (val < img[width * (y - 1) + x + 1]) ? 128 : 0;
		bi += (val < img[width * y + x + 1]) ? 128 : 0;
		bi += (val < img[width * (y + 1) + x + 1]) ? 128 : 0;
		bi += (val < img[width * (y + 1) + x]) ? 128 : 0;
		bi += (val < img [[width * (y + 1) + x - 1]) ? 128 : 0;
		bi += (val < img[width * y + x - 1]) ? 128 : 0;
		bi += (val < img[width * y + x - 1]) ? 128 : 0;
		for (int j = 0; j < 16; j++) {
			hist =
		}
	}
	

}
void main() {
	int* update_pos = (int*)calloc(4, sizeof(int));
	update_landmark_SDM(update_pos);
	printf("Landmark update\n");
	//printf("A : x = % d, y = % d, B : x = % d, y = % d\n", 코드 작성);
	int* update_pos = (int*)calloc(4, sizeof(int));
	int* LBP_img = (int*)calloc(48, sizeof(int));
	int* original_LBP_hist = (int*)calloc(256, sizeof(int));
	int* update_LBP_hist = (int*)calloc(256, sizeof(int));
	LBP_histogram(updata_pos, LBP_img, original_LBP_hist, update_LBP_hist);


}