#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat rotate(Mat src, double angle)   
{
    Mat dst;      
    Point2f pt(src.cols / 2., src.rows / 2.);            
    Mat r = getRotationMatrix2D(pt, angle, 1.0);      
    warpAffine(src, dst, r, Size(src.cols, src.rows));  
    return dst;         
}

Mat myRotate(Mat src, double angle) {
    Mat dst;
    dst = Mat::zeros(Size(src.cols, src.rows), CV_8UC3);
    int pt_x = src.cols / 2;
    int pt_y = src.rows / 2;
    int new_x;
    int new_y;
    double pi = 3.141592;
    double seta;

    seta = pi / (180.0 / angle);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            new_x = (i - pt_y) * sin(seta) + (j - pt_x) * cos(seta) + pt_x;
            new_y = (i - pt_y) * cos(seta) - (j - pt_x) * sin(seta) + pt_y;

            if (new_x < 0)		continue;
            if (new_x >= src.cols)	continue;
            if (new_y < 0)		continue;
            if (new_y >= src.rows)	continue;


            dst.at<Vec3b>(new_y, new_x)[0] = src.at<Vec3b>(i, j)[0];
            dst.at<Vec3b>(new_y, new_x)[1] = src.at<Vec3b>(i, j)[1];
            dst.at<Vec3b>(new_y, new_x)[2] = src.at<Vec3b>(i, j)[2];
        }
    }
    return dst;


}
void fnInterpolate(Mat img)
{
    int left_pixval[3] = { 0, };
    int right_pixval[3] = { 0. };


    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (j == 0) {
                right_pixval[0] = img.at<Vec3b>(i, j + 1)[0];
                right_pixval[1] = img.at<Vec3b>(i, j + 1)[1];
                right_pixval[2] = img.at<Vec3b>(i, j + 1)[2];
                left_pixval[0] = right_pixval[0];
                left_pixval[1] = right_pixval[1];
                left_pixval[2] = right_pixval[2];
            }
            else if (j == img.cols - 1) {
                left_pixval[0] = img.at<Vec3b>(i, j - 1)[0];
                left_pixval[1] = img.at<Vec3b>(i, j - 1)[1];
                left_pixval[2] = img.at<Vec3b>(i, j - 1)[2];
                right_pixval[0] = left_pixval[0];
                right_pixval[1] = left_pixval[1];
                right_pixval[2] = left_pixval[2];
            }
            else {
                left_pixval[0] = img.at<Vec3b>(i, j - 1)[0];
                left_pixval[1] = img.at<Vec3b>(i, j - 1)[1];
                left_pixval[2] = img.at<Vec3b>(i, j - 1)[2];
                right_pixval[0] = img.at<Vec3b>(i, j + 1)[0];
                right_pixval[1] = img.at<Vec3b>(i, j + 1)[1];
                right_pixval[2] = img.at<Vec3b>(i, j + 1)[2];
            }


            if ((img.at<Vec3b>(i, j)[0] == 0 && img.at<Vec3b>(i, j)[1] == 0 && img.at<Vec3b>(i, j)[2] == 0) &&
                (left_pixval[0] != 0 || left_pixval[1] != 0 || left_pixval[2] != 0) &&
                (right_pixval[0] != 0 || right_pixval[1] != 0 || right_pixval[2] != 0)) {
                img.at<Vec3b>(i, j)[0] = (left_pixval[0] + right_pixval[0]) / 2;
                img.at<Vec3b>(i, j)[1] = (left_pixval[1] + right_pixval[1]) / 2;
                img.at<Vec3b>(i, j)[2] = (left_pixval[2] + right_pixval[2]) / 2;
            }
        }
    }

   
}

int main()
{
    Mat src = imread("src.png");         
    double degree;
    cout << "Input your scale degree: ";
    cin >> degree;
    Mat dst;      
    dst = myRotate(src, -degree);      
    fnInterpolate(dst);
    imshow("src.png", src);         
    imshow("dst.png", dst);         
    waitKey(0);                    
    return 0;
}