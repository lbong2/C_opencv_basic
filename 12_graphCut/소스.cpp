#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/photo.hpp>
#include <cstdio>
#include <iostream>
using namespace cv;
using namespace std;

Mat markerMask, img;
Point prevPt(-1, -1);
static void onMouse(int event, int x, int y, int flags, void*)
{
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows)
        return;
    if ((event == EVENT_RBUTTONUP || !(flags & EVENT_FLAG_LBUTTON)) && (event == EVENT_LBUTTONUP  || !(flags & EVENT_FLAG_RBUTTON)))
        prevPt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN)
        prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))
    {
        Point pt(x, y);
        if (prevPt.x < 0)
            prevPt = pt;
        line(markerMask, prevPt, pt, 1, 5, 8, 0);
        line(img, prevPt, pt, Scalar(255, 0, 0), 5, 8, 0);
        prevPt = pt;
        imshow("image", img);
    }
    else if (event == EVENT_RBUTTONDOWN)
        prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_RBUTTON))
    {
        Point pt(x, y);
        if (prevPt.x < 0)
            prevPt = pt;
        line(markerMask, prevPt, pt, 0, 5, 8, 0);
        line(img, prevPt, pt, Scalar(0, 0, 255), 5, 8, 0);
        prevPt = pt;
        imshow("image", img);
    }
}
int main(int argc, char** argv)
{
    
    Mat img0 = imread("img.jpg", 1), imgGray;
    //resize(img0, img0, Size(540, 720));
    Rect r(80, 90, 500, 350);
    Mat bgd, fgd;
    if (img0.empty())
    {
        cout << "Couldn't open image ";
        return 0;
    }
   
    namedWindow("image", 1);
    img0.copyTo(img);
    cvtColor(img, markerMask, COLOR_BGR2GRAY);
  
    markerMask = Scalar::all(2);
    imshow("image", img);
    setMouseCallback("image", onMouse, 0);
    for (;;)
    {
        char c = (char)waitKey(0);
        if (c == 27)
            break;
        if (c == 'r')
        {
            grabCut(img0, markerMask, r, bgd, fgd, 1, GC_INIT_WITH_MASK);
            Mat foreground(img0.size(), CV_8UC3, Scalar(255, 255, 255));
            for (int i = 0; i < markerMask.cols; i++)
                for (int j = 0; j < markerMask.rows; j++) {
                    if (markerMask.at<uchar>(j, i) == 1 || markerMask.at<uchar>(j, i) == 3) {
                        markerMask.at<uchar>(j, i) = 255;
                    }
                    else
                        markerMask.at<uchar>(j, i) = 0;
                }
            img0.copyTo(foreground, markerMask);
            imshow("result", foreground);
            imwrite("mask.jpg", markerMask);

        }

    }
    return 0;
}