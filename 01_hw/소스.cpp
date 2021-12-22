#include "svm.h"

int main()
{
#if _DEBUG
    cout << "svmdigits.exe should be built as Release mode!" << endl;
    return 0;
#endif

    HOGDescriptor hog(Size(20, 20), Size(10, 10), Size(5, 5), Size(5, 5), 9);

    Ptr<SVM> svm = train_hog_svm(hog);

    if (svm.empty()) {
        cerr << "Training failed!" << endl;
        return -1;
    }
 


    CascadeClassifier cascade_km;
    cascade_km.load("C:\\Traffic_sig_fin\\fin\\kmSignal_fin.xml");
    CascadeClassifier cascade_st;
    cascade_st.load("C:\\Traffic_sig\\cascade\\stop\\stop_signal.xml");

    Mat img;
    vector<Rect> km;
    vector<Rect> stop;

    // video from cam
    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        printf("Couldn¡¯t open the web camera¡¦\n");
        return 0;
    }
    
    Mat digit = Mat::zeros(100, 100, CV_8UC3);
    while (1)
    {
        capture >> img;
        Rect bounds(0, 0, img.cols, img.rows);
        cascade_km.detectMultiScale(img, km, 1.1, 6, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));
        cascade_st.detectMultiScale(img, stop, 1.1, 6, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));
        for (int y = 0; y < km.size(); y++)
        {
            Point lb(km[y].x + km[y].width / 4, km[y].y + km[y].height / 4);
            Point tr(km[y].x + km[y].width / 2, km[y].y + km[y].height * 3 / 4);
            rectangle(img, lb, tr, Scalar(255, 0, 0), 3, 8, 0);
            Rect sq(km[y].x + km[y].width / 4, km[y].y + km[y].height / 4, km[y].width / 4, km[y].height / 2);
            digit = img(bounds & sq).clone();

            inRange(digit,  Scalar(0, 0, 0), Scalar(120, 120, 120), digit);

            resize(digit, digit, Size(40, 60), 0, 0, INTER_AREA);
            vector<float> desc;
            hog.compute(digit, desc);
            Mat desc_mat(desc);
            int res = cvRound(svm->predict(desc_mat.t()));
            cout << res << endl;
        }
        
        for (int y = 0; y < stop.size(); y++)
        {
            Point lb(stop[y].x + stop[y].width / 2, stop[y].y + stop[y].height);
            Point tr(stop[y].x, stop[y].y);
            rectangle(img, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
        }
        imshow("img", img);
        imshow("digit", digit);
        if (waitKey(1) == 27) {
            break;
        }
        
    }

    return 0;
}