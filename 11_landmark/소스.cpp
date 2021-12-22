#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"
#pragma warning (disable : 4996)

using namespace std;
using namespace cv;

FILE* fp;
void make_histo(Mat in, int x, int y, int k, float* histo) {
    float* X;
    float* Y;
    X = (float*)calloc(16 * 16, sizeof(float));
    Y = (float*)calloc(16 * 16, sizeof(float));

    if (x < 8 || y < 8 || x > in.cols - 10 || y > in.rows - 10) {
        return;
    }
        
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            int Xidx = x - 7 + i;
            int Yidx = y - 7 + j;
            X[j * 16 + i] = in.at<uchar>(Yidx, Xidx + 1) - in.at<uchar>(Yidx, Xidx - 1);
            Y[j * 16 + i] = in.at<uchar>(Yidx - 1, Xidx) - in.at<uchar>(Yidx + 1, Xidx);
        }
    }
    float mag;
    float mag_sum = 0.;
    float x_val, y_val;
    float idx = 0.;
    int histo_height = 0;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            x_val = X[j * 16 + i];
            y_val = Y[j * 16 + i];
            mag = sqrt(x_val * x_val + y_val * y_val);
            mag_sum += mag;
            idx = (atan2(y_val, x_val) >= 0) ? (atan2(y_val, x_val) * 57.29577951) : ((atan2(y_val, x_val) * 57.29577951) + 180);
            histo[(k - 27) * 9 + (int)(idx / 20.)] += mag;
        }
    }
    
    for (int h = 0; h < 9; h++) {
        //histo[(k - 27) * 9 + h] /= mag_sum + 1;
    }
    free(X);
    free(Y);

}
int main()
{
    float* ref_histo = (float*)calloc(40 * 9, sizeof(float));
    float* histo = (float*)calloc(40 * 9, sizeof(float));
    fp = fopen("histo.txt", "r");
    for (int i = 0; i < 40 * 9; i++) {
        fscanf(fp, "%f\n", ref_histo + i);
    }
    fclose(fp);
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "ÎÄ¼þ´ò¿ª´íÎó£¬ÇëÖØÐÂÊäÈëÎÄ¼þÂ·¾¶." << std::endl;
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera(0);
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    cv::Mat Image;
    Mat temp;
    cv::Mat current_shape;
    int correct;


    while (1) {
        for (int i = 0; i < 40 * 9; i++) {
            histo[i] = 0.;
        }
        correct = 0;
        mCamera >> Image;
        temp = Image.clone();
        cvtColor(temp, temp, COLOR_BGR2GRAY);
        modelt.track(Image, current_shape);
        cv::Vec3d eav;
        modelt.EstimateHeadPose(current_shape, eav);
        modelt.drawPose(Image, current_shape, 50);

        int numLandmarks = current_shape.cols / 2;
        for (int j = 27; j < numLandmarks; j++) {
            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);

            //È÷½ºÅä±×·¥ Â÷ÀÌ ´Ù ´õÇÑ°Å
            make_histo(temp, x, y, j, histo);
        }
        float differ = 0.;
        for (int i = 0; i < 40 * 9; i++) {
            differ += abs(histo[i] - ref_histo[i]);
        }
        printf("%f\n", differ);
        if (differ < 60000) {
            correct = 1;
        }
        for (int j = 27; j < numLandmarks; j++) {
            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);
            if (correct == 0) {
                //cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
                cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
            }
            else {
                //cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 255, 0));
                cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
            }
        }
        cv::imshow("Camera", Image);

        if (27 == cv::waitKey(5)) {
            
            break;
        }
        
    }

    
    return 0;
}
