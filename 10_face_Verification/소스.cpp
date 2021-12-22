#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;
using namespace cv::ml;
using namespace std;
//   1  1  1  1 1 1 1 1
// 128 64 32 16 8 4 2 1
static int uniform[256] =
{
	0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
	14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
	58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
	58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
	58,58,58,50,51,52,58,53,54,55,56,57
};

void cal_LBP(Mat in, Mat bi) {
	int val;

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
}
void make_histo(Mat en, Mat bi, float** histo) {
	int count = 0; // histogram 줄 표시용 
	int val;
	for (int y = 0; y <= en.rows - 32; y += 16) {
		for (int x = 0; x <= en.cols - 32; x += 16) {

			for (int j = 0; j < 32; j++) {
				for (int i = 0; i < 32; i++) {
					val = (int)bi.at<uchar>(y + j, x + i);
					histo[count][val] += 1.;
				}
			}
			count++;
		}
	}
	float histo_sum;
	
	// en histo normalization
	for (int i = 0; i < (en.rows / 16 - 1) * (en.cols / 16 - 1); i++) {
		histo_sum = 0.;
		for (int a = 0; a < 256; a++) {
			histo_sum += histo[i][a] * histo[i][a];
		}
		histo_sum = sqrt(histo_sum);
		for (int a = 0; a < 256; a++) {
			histo[i][a] /= histo_sum;
		}
	}
}
void main() {

	Mat en;
	
	en = imread("myface.jpg", IMREAD_GRAYSCALE);
	
	
	Mat loc_bi = Mat::zeros(en.rows, en.cols, CV_8UC1); // en LBP
	 
	cal_LBP(en, loc_bi);
	for (int i = 0; i < en.cols; i++) {
		for (int j = 0; j < en.rows; j++) {
			int temp = loc_bi.at<uchar>(j, i);
			loc_bi.at<uchar>(j, i) = uniform[temp];
		}
	}
	
	// en histo 동적 할당
	float** en_histo;
	en_histo = (float**)calloc((en.rows / 16 - 1) * (en.cols / 16 - 1), sizeof(float*));
	for (int i = 0; i < (en.rows / 16 - 1) * (en.cols / 16 - 1); i++) {
		en_histo[i] = (float*)calloc(256, sizeof(float));
	}

	// ver histo 동적 할당
	float** ver_histo;
	ver_histo = (float**)calloc((en.rows / 16 - 1) * (en.cols / 16 - 1), sizeof(float*));
	for (int i = 0; i < (en.rows / 16 - 1) * (en.cols / 16 - 1); i++) {
		ver_histo[i] = (float*)calloc(256, sizeof(float));
	}

	// en histo 구현, normalization 
	make_histo(en, loc_bi, en_histo);
	


	VideoCapture capture(0);
	Mat frame;
	Mat frame_gray;
	CascadeClassifier cascade;

	cascade.load("C:/Opencv/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
	if (!capture.isOpened()) {
		printf("Couldn’t open the web camera…\n");
		return;
	}
	float differ;
	while (true) {
		Mat loc_bi2 = Mat::zeros(en.rows, en.cols, CV_8UC1);// ver LBP
		capture >> frame;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		vector<Rect> faces;
		cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(10, 10));
		for (int y = 0; y < faces.size(); y++)
		{
			differ = 0.;
			Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
			Point tr(faces[y].x, faces[y].y);

			Rect bounds(0, 0, frame.cols, frame.rows); 
			Rect r(faces[y].x, faces[y].y, faces[y].width, faces[y].height);
			Mat face_crop = frame_gray(r & bounds);
			resize(face_crop, face_crop, Size(en.cols, en.rows), 0, 0, 1);
			// ver histo 구현 
			cal_LBP(face_crop, loc_bi2);
			for (int i = 0; i < en.cols; i++) {
				for (int j = 0; j < en.rows; j++) {
					int temp = loc_bi2.at<uchar>(j, i);
					loc_bi2.at<uchar>(j, i) = uniform[temp];
				}
			}
			make_histo(face_crop, loc_bi2, ver_histo);

			// en - ver
			for (int i = 0; i < (en.rows / 16 - 1) * (en.cols / 16 - 1); i++) {
				for (int a = 0; a < 256; a++) {
					differ += (en_histo[i][a] - ver_histo[i][a] > 0) ? en_histo[i][a] - ver_histo[i][a] : ver_histo[i][a] - en_histo[i][a];
				}
			}
			// printf("differ : %f", differ);
			if(differ < 195)
				rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
			else 
				rectangle(frame, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
			
		}
		imshow("Video", frame);
		if (waitKey(30) >= 0) break;
	}


	// 동적할당 해제
	for (int i = 0; i < 16; i++) {
		free(en_histo[i]);
		free(ver_histo[i]);
	}
	free(en_histo);
	free(ver_histo);
	
}