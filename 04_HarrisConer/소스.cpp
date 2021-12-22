#include<cstdio>
#include<cmath>
#include<cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#pragma warning(disable : 4996)
using namespace cv;
#define PI 3.141592
FILE* fp;
void Grad(int r, int c, float** X, float** Y, Mat in) {
	for (int i = 0; i < c; i++) {
		for (int j = 0; j < r; j++) {
			if (i == 0) {
				X[j][i] = (int)in.at<uchar>(j, i + 1);
			}
			else if (i == c - 1)
			{
				X[j][i] = -(int)in.at<uchar>(j, i - 1);
			}
			else
				X[j][i] = (int)in.at<uchar>(j, i + 1) - (int)in.at<uchar>(j, i - 1);
		}
	}
	for (int i = 0; i < c; i++) {
		for (int j = 0; j < r; j++) {
			if (j == 0) {
				Y[j][i] = -(int)in.at<uchar>(j + 1, i);
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
void cal_R(Mat ref, float** X, float** Y, float** R) {
	
	float k = 0.04f;
	float g[3][3] = {{1,2,1},
					{2,4,2},
					{1,2,1} };
	for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
		{
			g[y][x] /= 16.f;
		}

	float tx2, ty2, txy;

	for (int j = 1; j < ref.rows - 1; j++)
		for (int i = 1; i < ref.cols - 1; i++)
		{
			tx2 = ty2 = txy = 0;
			for (int y = 0; y < 3; y++){
				for (int x = 0; x < 3; x++)
				{
					tx2 += (X[j + y - 1][i + x - 1] * X[j + y - 1][i + x - 1] * g[y][x]);
					ty2 += (Y[j + y - 1][i + x - 1] * Y[j + y - 1][i + x - 1] * g[y][x]);
					txy += (X[j + y - 1][i + x - 1] * Y[j + y - 1][i + x - 1] * g[y][x]);
				}

			R[j][i] = (tx2 * ty2 - txy * txy)
				- k * (tx2 + ty2) * (tx2 + ty2);
			}
				
		}
	
}

void keypoint_matching() {
	Mat src1 = imread("ref.bmp", IMREAD_GRAYSCALE);
	Mat src2 = imread("tar.bmp", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty()) {
		cerr << "Image load failed!" << endl;
		return;
	}
	Ptr<Feature2D> feature = ORB::create();

	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	std::sort(matches.begin(), matches.end());
	vector < DMatch> good_matches(matches.begin(), matches.begin() + 50);

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}
void main() {
	Mat ref = imread("ref.bmp", CV_8UC1);
	Mat tar = imread("tar.bmp", CV_8UC1);
	Mat result = imread("ref.bmp", IMREAD_COLOR);
	Mat result2 = imread("tar.bmp", IMREAD_COLOR);
	Mat fin_img = Mat(ref.rows, ref.cols * 2, CV_8UC1);

	float** ref_X, ** ref_Y, ** ref_R;
	float** tar_X, ** tar_Y, ** tar_R;
	ref_X = (float**)calloc(ref.rows, sizeof(float*));
	ref_Y = (float**)calloc(ref.rows, sizeof(float*));
	ref_R = (float**)calloc(ref.rows, sizeof(float*));


	tar_X = (float**)calloc(tar.rows, sizeof(float*));
	tar_Y = (float**)calloc(tar.rows, sizeof(float*));
	tar_R = (float**)calloc(tar.rows, sizeof(float*));


	for (int i = 0; i < ref.rows; i++) {
		ref_X[i] = (float*)calloc(ref.cols, sizeof(float));
		ref_Y[i] = (float*)calloc(ref.cols, sizeof(float));
		ref_R[i] = (float*)calloc(ref.cols, sizeof(float));

	}
	for (int i = 0; i < tar.rows; i++) {
		tar_X[i] = (float*)calloc(tar.cols, sizeof(float));
		tar_Y[i] = (float*)calloc(tar.cols, sizeof(float));
		tar_R[i] = (float*)calloc(tar.cols, sizeof(float));
	}

	Grad(ref.rows, ref.cols, ref_X, ref_Y, ref);
	cal_R(ref, ref_X, ref_Y, ref_R);

	Grad(tar.rows, tar.cols, tar_X, tar_Y, tar);
	cal_R(tar, tar_X, tar_Y, tar_R);
	

	Scalar c;
	Point pCenter;
	c.val[0] = 0;
	c.val[1] = 0;
	c.val[2] = 255;
	int radius = 1;
	for (int i = 1; i < ref.cols - 1; i++) {
		for (int j = 1; j < ref.rows - 1; j++) {
			int th1 = ref_R[j][i];
			int th2 = tar_R[j][i];
			if (th1 > 200000) {

				pCenter.x = i;
				pCenter.y = j;
				circle(result, pCenter, radius, c, 2, 8, 0); // red circle
			}
			if (th2 > 200000) {
				pCenter.x = i;
				pCenter.y = j;
				circle(result2, pCenter, radius, c, 2, 8, 0); // red circle
			}
		}
	}
	for (int i = 1; i < tar.cols - 1; i++) {
		for (int j = 1; j < tar.rows - 1; j++) {
			
		}
	}
	imshow("result", result);
	imshow("result2", result2);
	waitKey(0);
	keypoint_matching();
	

	for (int i = 0; i < ref.rows; i++) {
		free(ref_X[i]);
		free(ref_Y[i]);
		free(ref_R[i]);
	}
	for (int i = 0; i < tar.rows; i++) {
		free(tar_X[i]);
		free(tar_Y[i]);
		free(tar_R[i]);
	}
	free(ref_X);
	free(ref_Y);
	free(ref_R);
	free(tar_X);
	free(tar_Y);
	free(tar_R);

}