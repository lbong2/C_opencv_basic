#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace std;

#define pi 3.141592653589793238462643383279

void imagecheck(double** input, int width, int height, double scale, char* imgname)
{
	// output Mat 생성
	Mat output(height, width, CV_8UC1);
	// ResultImage에 result 포인터 배열 삽입
	for (int r = 0; r < height; ++r)
		for (int c = 0; c < width; ++c)
			output.at<uchar>(r, c) = input[r][c];
	// 크기 변경을 위한 Mat 생성
	Mat imgResize;
	resize(output, imgResize, Size(width * scale, height * scale));
	imshow(imgname, imgResize);
}

void GaussianFilter(double** input, double** ginput, int width, int height)
{
	int i, j, mc, mr;
	double sigma = 1.0;

	for (j = 0; j < height; j++) {
		for (i = 0; i < width; i++) {
			double temp = 0;
			for (mc = 0; mc < 5; mc++) {
				for (mr = 0; mr < 5; mr++) {
					int maskwidth = i + (mr - 2);
					int maskheight = j + (mc - 2);
					if (maskwidth < 0 || maskheight < 0 || maskwidth > width || maskheight > height)
						continue;
					else
					{
						temp += (1 / (sqrt(2.0 * pi) * sigma))
							* exp(-((input[maskheight][maskwidth] - input[j][i]) * (input[maskheight][maskwidth] - input[j][i])) / (2.0 * sigma * sigma));
					}
				}
			}
			printf("%d %d ", i, j);
			ginput[j][i] += temp;
			printf("%.f\n", ginput[j][i]);
		}
	}
}

void CornerDetect(double** input, double** R_value, double** dx2, double** dy2, double** dxy, int width, int height)
{
	double tx, ty;
	for (int j = 1; j < height - 1; j++)
		for (int i = 1; i < width - 1; i++)
		{
			tx = (input[j - 1][i + 1] + input[j][i + 1] + input[j + 1][i + 1]
				- input[j - 1][i - 1] - input[j][i - 1] - input[j + 1][i - 1]) / 6.f;

			ty = (input[j + 1][i - 1] + input[j + 1][i] + input[j + 1][i + 1]
				- input[j - 1][i - 1] - input[j - 1][i] - input[j - 1][i + 1]) / 6.f;

			dx2[j][i] = tx * tx;
			dy2[j][i] = ty * ty;
			dxy[j][i] = tx * ty;
		}

	// R_value 포인터 배열 = R = det(M) - k*(tr(M))**2
	float k = 0.04f;
	for (int j = 2; j < height - 2; j++)
		for (int i = 2; i < width - 2; i++)
		{
			R_value[j][i] = (dx2[j][i] * dy2[j][i] - dxy[j][i] * dxy[j][i])
				- k * (dx2[j][i] + dy2[j][i]) * (dx2[j][i] + dy2[j][i]);
		}
}

void CornerPoint(double** input, double** R_value, int width, int height)
{
	int radius = 3;
	Point pCenter;
	Scalar c;
	c.val[0] = 0;
	c.val[1] = 0;
	c.val[2] = 255;

	// output Mat 생성
	Mat output(height, width, CV_8UC3);
	// ResultImage에 input 포인터 배열 삽입
	for (int r = 0; r < height; ++r)
		for (int c = 0; c < width; ++c) {
			output.at<Vec3b>(r, c)[0] = input[r][c];
			output.at<Vec3b>(r, c)[1] = input[r][c];
			output.at<Vec3b>(r, c)[2] = input[r][c];
		}
	Mat img_circle;
	output.copyTo(img_circle);
	for (int j = 2; j < height - 2; j++)
		for (int i = 2; i < width - 2; i++)
		{
			if (R_value[j][i] > 0)
			{
				pCenter.x = i;
				pCenter.y = j;
				circle(output, pCenter, radius, c, 2, 8, 0);
			}
		}

	imshow("output", output);
}

void main()
{
	Mat ref = imread("ref.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat tar = imread("tar.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	const int width = ref.cols; // 행길이 계산
	const int height = ref.rows; // 열길이 계산

	// input,  dx2, dy2, dxy 배열 생성
	double** input = new double* [height];
	double** dx2 = new double* [height];
	double** dy2 = new double* [height];
	double** dxy = new double* [height];

	// 가우시안 배열 생성
	double** ginput = new double* [height];

	// output 배열 생성
	double** R_value = new double* [height];
	
	for (int r = 0; r < height; ++r)
	{
		input[r] = new double[width];
		dx2[r] = new double[width];
		dy2[r] = new double[width];
		dxy[r] = new double[width];
		ginput[r] = new double[width];
		R_value[r] = new double[width];
	}

	// input 포인터 배열에 ref, tar 삽입
	for (int r = 0; r < height; ++r) {
		for (int c = 0; c < width; ++c) {
			input[r][c] = ref.at<uchar>(r, c);
		}
		//for (int c = width; c < width * 2; ++c) {
		//	input[r][c] = tar.at<uchar>(r, c);
		//}
	}

	//GaussianFilter(input, ginput, width, height);

	// Harris Corner Detect
	CornerDetect(input, R_value, dx2, dy2, dxy, width, height);

	// Corner point
	CornerPoint(input, R_value, width, height);

	// 이미지 출력
	char imgname1[] = "ginput";
	char* p1;
	p1 = imgname1;
	imagecheck(ginput, width, height, 1.0, imgname1);
	waitKey(0);
}