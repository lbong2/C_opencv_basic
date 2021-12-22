#include<opencv2/imgproc.hpp>
#include<opencv2/photo.hpp>
#include<opencv2/highgui.hpp>
#include<stdlib.h>
#include<iostream>
#include<stdio.h>
#pragma warning(disable: 4996)


using namespace cv;
using namespace std;
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

/// <summary>
/// Gradient 구하는 함수
/// </summary>
/// <param name="in">: 입력 이미지 </param>
/// <param name="r">: 입력 이미지 세로 길이 </param>
/// <param name="c">: 입력 이미지 가로 길이 </param>
/// <param name="X">: X방향 Gradient 저장 </param>
/// <param name="Y">: Y방향 Gradient 저장 </param>
void Grad(Mat in, int r, int c, float* X, float* Y) {
	cvtColor(in, in, COLOR_BGR2GRAY);
	for (int j = 0; j < r; j++) { // i = x, j = y
		for (int i = 0; i < c; i++) {
			if (i == 0) {
				if (j == 0) X[j * c + i] = in.at<uchar>(j, i + 1) + in.at<uchar>(j + 1, i + 1);
				else if (j == r - 1) X[j * c + i] = in.at<uchar>(j, i + 1) + in.at<uchar>(j - 1, i + 1);
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
/// <summary>
/// Histogram 구축 함수
/// </summary>
/// <param name="in">: 입력 이미지 </param>
/// <param name="r">: 입력 이미지 세로 길이 </param>
/// <param name="c">: 입력 이미지 가로 길이 </param>
/// <param name="X">: X방향 Gradient 저장 </param>
/// <param name="Y">: Y방향 Gradient 저장 </param>
/// <param name="histo">: Histogram 저장 </param>
void make_histo(Mat in, int r, int c, float* X, float* Y, float* histo) {
	int mag;
	int mag_sum;
	float x_val, y_val;
	double idx;
	int histo_height = 0;
	for (int i = 0; i <= c - 16; i += 8) {
		for (int j = 0; j <= r - 16; j += 8) {
			mag_sum = 0;
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++) {
					x_val = X[(j + y) * c + i + x];
					y_val = Y[(j + y) * c + i + x];
					mag = sqrt(x_val * x_val + y_val * y_val);
					mag_sum += mag;
					idx = (atan2(y_val, x_val) >= 0) ? (atan2(y_val, x_val) * 57.29577951) : ((atan2(y_val, x_val) * 57.29577951) + 180);
					histo[histo_height * 9 + (int)(idx / 20.)] += mag;
				}
			}
			for (int h = 0; h < 9; h++) {
				histo[histo_height * 9 + h] /= mag_sum * 10 + 1;
			}

			histo_height++;

		}
	}


}
/// <summary>
/// HOG Descriptor wrapper 함수
/// </summary>
/// <param name="in">: 입력 이미지 </param>
/// <param name="X">: X방향 Gradient 저장 </param>
/// <param name="Y">: Y방향 Gradient 저장 </param>
/// <param name="histo">: Histogram 저장 </param>
void HOG_descriptor(Mat in, float* X, float* Y, float* histo) {
	int r = in.rows;
	int c = in.cols;
	
	Grad(in, r, c, X, Y);
	make_histo(in, r, c, X,  Y, histo);
}

/// <summary>
/// 픽셀별 R값 계산
/// </summary>
/// <param name="in">: 입력 이미지</param>
/// <param name="X">: X방향 Gradient 저장 </param>
/// <param name="Y">: Y방향 Gradient 저장 </param>
/// <param name="R">: R값 저장</param>
void cal_R(Mat in, float* X, float* Y, float* R) {

	for (int i = 0; i < in.cols; i++)
		for (int j = 0; j < in.rows; j++) {
			X[j * in.cols + i] /= 10.;
			Y[j * in.cols + i] /= 10.;
		}
	float k = 0.04f;
	float g[3][3] = { {1,2,1},
					{2,4,2},
					{1,2,1} };
	for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
		{
			g[y][x] /= 16.f;
		}

	float tx2, ty2, txy;

	for (int j = 1; j < in.rows - 1; j++)
		for (int i = 1; i < in.cols - 1; i++)
		{
			tx2 = ty2 = txy = 0;
			for (int y = 0; y < 3; y++) {
				for (int x = 0; x < 3; x++)
				{
					tx2 += (X[(j + y - 1) * in.cols + i + x - 1] * X[(j + y - 1) * in.cols + i + x - 1] * g[y][x]);
					ty2 += (Y[(j + y - 1) * in.cols + i + x - 1] * Y[(j + y - 1) * in.cols + i + x - 1] * g[y][x]);
					txy += (X[(j + y - 1) * in.cols + i + x - 1] * Y[(j + y - 1) * in.cols + i + x - 1] * g[y][x]);
				}

				R[j * in.cols + i] = (tx2 * ty2 - txy * txy)
					- k * (tx2 + ty2) * (tx2 + ty2);
			}

		}

}
/// <summary>
/// Corner Map을 형성하는 함수
/// </summary>
/// <param name="in">: 입력 이미지</param>
/// <param name="R">: R값 저장</param>
/// <param name="th">: Threshold 값 </param>
/// <returns>: Corner Map(GrayScale)</returns>
Mat cornerMap(Mat in, float* R, int th) {
	Mat out = Mat::zeros(in.rows, in.cols, CV_8UC1);
	for (int i = 1; i < in.cols - 1; i++) {
		for (int j = 1; j < in.rows - 1; j++) {
			if (R[j * in.cols + i] > th) {
				out.at<uchar>(j, i) = 255;
			}
		}
	}
	return out;
}
/// <summary>
/// Corner지점에 빨간 동그라미를 그린 이미지 반환
/// </summary>
/// <param name="in">: 입력 이미지</param>
/// <param name="R">: R값 저장</param>
/// <param name="th">: Threshold 값 </param>
/// <returns>: Circle Map(COLOR)</returns>
Mat cornerCircle(Mat in, float* R, int th) {
	Mat out = Mat::zeros(in.rows, in.cols, CV_8UC3);
	out = in.clone();
	
	Point pCenter;
	
	int radius = 1;
	for (int i = 1; i < in.cols - 1; i++) {
		for (int j = 1; j < in.rows - 1; j++) {
			if (R[j * in.cols + i] > th) {
				pCenter.x = i;
				pCenter.y = j;
				circle(out, pCenter, radius, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	return out;
}
/// <summary>
/// 
/// </summary>
/// <param name="in">: 입력 이미지 </param>
/// <param name="r">: 입력 이미지 세로 길이 </param>
/// <param name="c">: 입력 이미지 가로 길이 </param>
/// <param name="X">: X방향 Gradient 저장 </param>
/// <param name="Y">: Y방향 Gradient 저장 </param>
/// <param name="R">: R값 저장</param>
/// <param name="th">: Threshold 값 </param>
/// <param name="flag">
/// 0 : make cornerCircle 
/// 1 : make cornerMap</param>
/// <returns>: 출력 이미지 </returns>
Mat HarrisCornerDetect(Mat in, int r, int c, float* X, float* Y, float* R, int th = 5000, int flag = 0) {
	Mat out;
	Grad(in, r, c, X, Y);
	cal_R(in, X, Y, R);
	if(flag == 0)
		out = cornerCircle(in, R, th);
	else 
		out = cornerMap(in,  R, th);
	return out;
}
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
/// <summary>
/// 2개 이미지의 LBP Histogram을 비교한 값을 반환 하는 함수
/// </summary>
/// <param name="img1">: 이미지 1</param>
/// <param name="img2">: 이미지 2</param>
/// <param name="histo1">: 이미지 1의  histo</param>
/// <param name="histo2">: 이미지 2의 histo</param>
/// <param name="uni_flag">: 유니폼 적용 여부</param>
/// <returns>:LBP Histogram 차이 </returns>
float compLBP(Mat img1, Mat img2, float** histo1, float** histo2, int uni_flag = 0) {
	Mat bi1, bi2;
	bi1 = cal_LBP(img1, uni_flag);
	bi2 = cal_LBP(img2, uni_flag);
	makeLBPhisto(bi1, histo1);
	makeLBPhisto(bi2, histo2);
	float differ = 0.;
	for (int i = 0; i < (img1.rows / 16 - 1) * (img1.cols / 16 - 1); i++) {
		for (int a = 0; a < 256; a++) {
			differ += (histo1[i][a] - histo2[i][a] > 0) ? histo1[i][a] - histo2[i][a] : histo2[i][a] - histo1[i][a];
		}
	}
	return differ;
}
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
	Mat ref = imread("face_ref.bmp", IMREAD_COLOR);
	Mat tar = imread("face_tar.bmp", IMREAD_COLOR);
	Mat result = imread("face_tar.bmp", IMREAD_COLOR);
	Mat window;
	float** ref_histo, ** tar_histo;
	ref_histo = (float**)calloc(1, sizeof(float*));
	ref_histo[0] = (float*)calloc(256, sizeof(float));
	tar_histo = (float**)calloc(1, sizeof(float*));
	tar_histo[0] = (float*)calloc(256, sizeof(float));
	Rect bounds(0, 0, tar.cols, tar.rows);
	float differ;
	for (int j = 0; j <= tar.rows - 32; j++) {
		for (int i = 0; i <= tar.cols - 32; i++) {
			for (int a = 0; a < 256; a++) {
				ref_histo[0][a] = 0.;
				tar_histo[0][a] = 0.;
			}
			differ = 0.;
			Rect sq(i, j, 32, 32);
			window = tar(bounds & sq).clone();
			
			
			differ = compLBP(ref, window, ref_histo, tar_histo, 1);
			if (differ < 1.) {
				rectangle(result, Rect(i, j, 32, 32), Scalar(0, 0, 255));
				printf("%d, %d : %f\n", i, j, differ);
			}
			
			
		}
	}
	imshow("result", result);
	waitKey(0);

	
}
// resize 
/*Mat img = imread("Lenna.png", IMREAD_COLOR);
	Mat result;
	printf("width : %d\t height : %d\n", img.cols, img.rows);
	result = Mat(300, 400, CV_8UC3);
	myResize(img, img, 400, 300);
	printf("width : %d\t height : %d\n", img.cols, img.rows);

	imshow("before", img);
	waitKey(0);*/

// HOG Descriptor
/*Mat img = imread("Lenna.png", IMREAD_COLOR);
	float* X, * Y, * histo;
	X = (float*)calloc(img.cols * img.rows, sizeof(float));
	Y = (float*)calloc(img.cols * img.rows, sizeof(float));
	histo = (float*)calloc((img.cols / 8 - 1) * (img.rows / 8 - 1) * 9, sizeof(float));
	HOG_descriptor(img, X, Y, histo);
	printf("dd");*/

// Harris Corner Detect
/*Mat img = imread("Overay_Text.jpg", IMREAD_COLOR);
float* X = (float*)calloc(img.cols * img.rows, sizeof(float));
float* Y = (float*)calloc(img.cols * img.rows, sizeof(float));
float* R = (float*)calloc(img.cols * img.rows, sizeof(float));
Mat result = HarrisCornerDetect(img, img.rows, img.cols, X, Y, R, 5000, 1);
imshow("result", result);
waitKey(0);*/

// LBP Histogram
/*	Mat img1 = imread("img_1.jpg", IMREAD_COLOR);
	Mat img2 = imread("img_2.jpg", IMREAD_COLOR);
	Mat img3 = imread("img_3.jpg", IMREAD_COLOR);
	
	resize(img1, img1, Size(160, 160));
	resize(img2, img2, Size(160, 160));
	resize(img3, img3, Size(160, 160));
	float** histo1 = (float**)calloc((img1.rows / 16 - 1) * (img1.cols / 16 - 1), sizeof(float*));
	for (int i = 0; i < (img1.rows / 16 - 1) * (img1.cols / 16 - 1); i++) {
		histo1[i] = (float*)calloc(256, sizeof(float));
	}
	float** histo2 = (float**)calloc((img1.rows / 16 - 1) * (img1.cols / 16 - 1), sizeof(float*));
	for (int i = 0; i < (img1.rows / 16 - 1) * (img1.cols / 16 - 1); i++) {
		histo2[i] = (float*)calloc(256, sizeof(float));
	}
	float** histo3 = (float**)calloc((img1.rows / 16 - 1) * (img1.cols / 16 - 1), sizeof(float*));
	for (int i = 0; i < (img1.rows / 16 - 1) * (img1.cols / 16 - 1); i++) {
		histo3[i] = (float*)calloc(256, sizeof(float));
	}
	printf("%f\n", compLBP(img1, img2, histo1, histo2, 1));
	printf("%f\n", compLBP(img1, img3, histo1, histo3, 1));*/