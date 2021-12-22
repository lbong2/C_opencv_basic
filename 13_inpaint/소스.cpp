#include "opencv2/imgproc.hpp"
#include <opencv2/photo.hpp>
#include "opencv2/highgui.hpp"
#include<stdlib.h>
#include <stack>
#include<iostream>

using namespace cv;
using namespace std;
void Grad(Mat in, int r, int c, float* X, float* Y);
int Labeling(Mat src, Mat dst);
void main() {
	
	Mat img = imread("Overay_text.jpg", IMREAD_COLOR), gray;
    Mat dst;
	Mat cornerMap, find_or, lable_text, final_text, final_x, final_y;
	int cornerCnt = 0;
	int max_den = 0;
	int max_idx = 0;
	// grayscale 이미지 만들기
	cvtColor(img, gray, COLOR_BGR2GRAY);
	cvtColor(img, cornerMap, COLOR_BGR2GRAY);
	cvtColor(img, find_or, COLOR_BGR2GRAY);
	cvtColor(img, lable_text, COLOR_BGR2GRAY);
	cvtColor(img, final_x, COLOR_BGR2GRAY);
	cvtColor(img, final_y, COLOR_BGR2GRAY);
	cvtColor(img, final_text, COLOR_BGR2GRAY);
	cornerMap = Scalar::all(0);
	find_or = Scalar::all(0);
	lable_text = Scalar::all(0);
	final_x = Scalar::all(0);
	final_y = Scalar::all(0);
	final_text = Scalar::all(0);
	img.copyTo(dst);
	// grad 저장할 동적할당 변수 
	float* gradX, *gradY, *R;
	gradX = (float*)calloc(img.cols * img.rows, sizeof(float));
	gradY = (float*)calloc(img.cols * img.rows, sizeof(float));
	R = (float*)calloc(img.cols * img.rows, sizeof(float));
	Grad(gray, img.rows, img.cols, gradX, gradY);

	// normalize
	
	for(int i = 0; i < img.cols; i++)
		for (int j = 0; j < img.rows; j++) {
			gradX[j * img.cols + i] /= 10.;
			gradY[j * img.cols + i] /= 10.;
		}
	// 가우시안필터링
	float g[3][3] = { {1,2,1},
					{2,4,2},
					{1,2,1} };
	for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
		{
			g[y][x] /= 16.f;
		}
	// 윈도우 사용 R값 추출
	float k = 0.04f;
	float tx2, ty2, txy;
	for (int j = 1; j < img.rows - 1; j++)
		for (int i = 1; i < img.cols - 1; i++)
		{
			tx2 = ty2 = txy = 0;
			for (int y = 0; y < 3; y++) {
				for (int x = 0; x < 3; x++)
				{
					tx2 += (gradX[(j + y - 1) * img.cols + i + x - 1] * gradX[(j + y - 1) * img.cols + i + x - 1] * g[y][x]);
					ty2 += (gradY[(j + y - 1) * img.cols + i + x - 1] * gradY[(j + y - 1) * img.cols + i + x - 1] * g[y][x]);
					txy += (gradX[(j + y - 1) * img.cols + i + x - 1] * gradY[(j + y - 1) * img.cols + i + x - 1] * g[y][x]);
				}

				R[j *img.cols + i] = (tx2 * ty2 - txy * txy)
					- k * (tx2 + ty2) * (tx2 + ty2);
			}

		}
	// 코너맵 만들기
	for (int i = 1; i < img.cols - 1; i++)
		for (int j = 1; j < img.rows - 1; j++) {
			//printf("%f\n", R[j * img.cols + i]);
			if (R[j * img.cols + i] > 200) {
				cornerMap.at<uchar>(j, i) = 255;
			}
		}
	imshow("corner", cornerMap);
	waitKey(0);

	// 임계값 넘는 윈도우 전체 흰색으로 색칠
	for (int j = 0; j < img.rows - 10; j++)
		for (int i = 0; i < img.cols - 10; i++) {
			cornerCnt = 0;
			for (int y = 0; y < 10; y++)
				for (int x = 0; x < 10; x++) {
					if (cornerMap.at<uchar>(j + y, i + x) == 255)
						cornerCnt++;

				}
			if (cornerCnt > 30) {
				for (int y = 0; y < 10; y++)
					for (int x = 0; x < 10; x++)
						find_or.at<uchar>(j + y, i + x) = 255;
			}
			cornerCnt = 0;
		}
	imshow("find_or", find_or);
	waitKey(0);
	// 레이블링 
	int Lnum = Labeling(find_or, lable_text);
	int *find_major;
	find_major = (int*)calloc(Lnum, sizeof(int));
	imshow("lable", lable_text);
	waitKey(0);
	// major num 찾기 
	for (int j = 0; j < img.rows; j++)
		for (int i = 0; i < img.cols; i++) {
			if(lable_text.at<uchar>(j, i) != 0)
				find_major[lable_text.at<uchar>(j, i)]++;
		}
	for (int i = 0; i < Lnum; i++) {
		if (max_den < find_major[i]) {
			max_idx = i;
		}
	}
	printf("%d\n", max_idx);
	for (int j = 0; j < img.rows; j++)
		for (int i = 0; i < img.cols; i++) {
			if (lable_text.at<uchar>(j, i) == max_idx - 1)
				lable_text.at<uchar>(j, i) = 255;
			else if (lable_text.at<uchar>(j, i) == 0) {}

			else lable_text.at<uchar>(j, i) = 0;
		}
	imshow("lable", lable_text);
	waitKey(0);
	for (int j = 0; j < img.rows; j++)
	{
		int sum = 0;
		for (int i = 0; i < img.cols; i++) {
			sum += lable_text.at<uchar>(j, i);
		}
		if(sum < img.cols * 255 * 0.15)
			for (int i = 0; i < img.cols; i++) {
				lable_text.at<uchar>(j, i) = 0;
			}
	}
	imshow("lable", lable_text);
	waitKey(0);
	for (int j = 0; j < img.rows; j++){
		for (int i = 0; i < img.cols; i++) {
			if (lable_text.at<uchar>(j, i) != 255) {
				cornerMap.at<uchar>(j, i) = 0;
			}
		}
	}
	imshow("text", cornerMap);
	waitKey(0);
	int st, fin;
	for (int j = 0; j < img.rows; j++) {
		st = fin = 0;
		for (int i = 0; i < img.cols; i++) {
			if (cornerMap.at<uchar>(j, i) == 255) {
				st = i;
				fin = i;
				for (int x = img.cols - 1; x >= 0; x--) {
					if (cornerMap.at<uchar>(j, x) == 255) {
						fin = x;
						break;
					}
				}
				for (int x = st; x <= fin; x++)
					final_x.at<uchar>(j, x) = 255;
				break;
			}
			
		}
	}
	for (int i = 0; i < img.cols; i++) {
		st = fin = 0;
		for (int j = 0; j < img.rows; j++) {
			if (cornerMap.at<uchar>(j, i) == 255) {
				st = j;
				fin = j;
				for (int y = img.rows - 1; y >= 0; y--) {
					if (cornerMap.at<uchar>(y, i) == 255) {
						fin = y;
						break;
					}
				}
				for (int y = st; y <= fin; y++)
					final_y.at<uchar>(y, i) = 255;
				break;
			}
		}
	}
	for (int j = 0; j < img.rows; j++) {
		for (int i = 0; i < img.cols; i++) {
			if (final_x.at<uchar>(j, i) ==255 || final_y.at<uchar>(j, i) == 255)
				final_text.at<uchar>(j, i) = 255;
		}
	}
	imshow("text", final_text);
	waitKey(0);
	for (int j = 0; j < img.rows; j++)
		for (int i = 0; i < img.cols; i++) {
			if (final_text.at<uchar>(j, i) == 255) {
				dst.at<Vec3b>(j, i)[0] = 255;
				dst.at<Vec3b>(j, i)[1] = 255;
				dst.at<Vec3b>(j, i)[2] = 255;
			}
		}
	inpaint(dst, final_text, dst, 10, INPAINT_TELEA);
	imshow("final", dst);
	waitKey(0);
	free(gradX);
	free(gradY);
	free(R);
}
int Labeling(Mat src, Mat dst)
{
	
	//// 8-neighbor labeling
	// Labeling 과정에서 stack overflow 방지를 위한 stl <stack>사용 
	stack<Point> st;
	int labelNumber = 0;
	for (int y = 1; y < src.rows - 1; y++) {
		for (int x = 1; x < src.cols - 1; x++) {
			// source image가 255일 경우 + Labeling 수행되지 않은 픽셀에서만 labeling process 시작
			if (src.at<uchar>(y, x) != 255 || dst.at<uchar>(y, x) != 0) continue;

			labelNumber++;

			// 새로운 label seed를 stack에 push
			st.push(Point(x, y));

			// 해당 label seed가 labeling될 때(stack이 빌 때) 까지 수행
			while (!st.empty()) {
				// stack top의 label point를 받고 pop
				int ky = st.top().y;
				int kx = st.top().x;
				int ny;
				int nx;
				st.pop();

				// label seed의 label number를 result image에 저장
				src.at<uchar>(ky, kx) = labelNumber;

				// search 4-neighbor
				ny = ky - 1;
				nx = kx;
				if (ny >= 0 && ny < src.rows)
					if (nx >= 0 && nx < src.cols) {
						if (src.at<uchar>(ny, nx) == 255 && dst.at<uchar>(ny, nx) == 0)
							st.push(Point(nx, ny));
						dst.at<uchar>(ny, nx) = labelNumber;
					}
				ny = ky;
				nx = kx + 1;
				if (ny >= 0 && ny < src.rows)
					if (nx >= 0 && nx < src.cols) {
						if (src.at<uchar>(ny, nx) == 255 && dst.at<uchar>(ny, nx) == 0)
							st.push(Point(nx, ny));
						dst.at<uchar>(ny, nx) = labelNumber;
					}
				ny = ky + 1;
				nx = kx;
				if (ny >= 0 && ny < src.rows)
					if (nx >= 0 && nx < src.cols) {
						if (src.at<uchar>(ny, nx) == 255 && dst.at<uchar>(ny, nx) == 0)
							st.push(Point(nx, ny));
						dst.at<uchar>(ny, nx) = labelNumber;
					}
				ny = ky;
				nx = kx - 1;
				if (ny >= 0 && ny < src.rows)
					if (nx >= 0 && nx < src.cols) {
						if (src.at<uchar>(ny, nx) == 255 && dst.at<uchar>(ny, nx) == 0)
							st.push(Point(nx, ny));
						dst.at<uchar>(ny, nx) = labelNumber;
					}
				
			}
		}
	}
	return labelNumber;
}
void Grad(Mat in, int r, int c, float* X, float *Y) {
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