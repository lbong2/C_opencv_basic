#include "LL.hpp"
/// <summary>
/// 
/// </summary>

void Grad(Mat, int, int, float*, float*);
void make_histo(Mat, int, int, float*, float*, float* histo);
void main()
{
	string motion[3];
	motion[0] = "rock";
	motion[1] = "scissor";
	motion[2] = "paper";
	VideoCapture cap(1);
	int i = 0;
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 960);
	Point setP(WIDTH * 0.7, HEIGHT * 0.3);
	Rect bounds(0, 0, WIDTH, HEIGHT);
	int frameCnt = 0;

	Mat frame;
	Mat oriR, oriS, oriP;
	Mat rock, scis, pap;
	Mat temp;
	Rect sq(setP.x + 1, setP.y + 1, SQ, SQ);
	if (!cap.isOpened()) {
		printf("Couldn¡¯t open the web camera¡¦\n");
		return;
	}

	while (true) {
		cap >> frame;
		resize(frame, frame, Size(WIDTH, HEIGHT));
		string myText = "Please locate your Rock in square (and Press 'C' Key)"; 
		circle(frame, Point(10, 40), 1, Scalar(255, 0, 0));
		putText(frame, myText, Point(10, 40), 2, 1.2, Scalar(0, 0, 255));
		rectangle(frame, Rect(setP, Point(setP.x + SQ + 2, setP.y + SQ + 2)), Scalar(0, 0, 255), 1, 8, 0);
		imshow("Video", frame);
		if (waitKey(60) == 27) {
			return;
		}
		if ((char)waitKey(60) == 'c') {
			rock = frame(bounds & sq).clone();
			break;
		}
	}
	waitKey(10);
	while (true) {
		cap >> frame;
		resize(frame, frame, Size(WIDTH, HEIGHT));
		string myText = "Please locate your Scissor in square (and Press 'C' Key)"; // + in 5s ~ 4s ~ 3s
		putText(frame, myText, Point(10, 40), 2, 1.2, Scalar(0, 0, 255));
		rectangle(frame, Rect(setP, Point(setP.x + SQ + 2, setP.y + SQ + 2)), Scalar(0, 0, 255), 1, 8, 0);
		imshow("Video", frame);
		if (waitKey(60) == 27) {
			return;
		}
		if ((char)waitKey(60) == 'c') {
			scis = frame(bounds & sq).clone();
			break;
		}
	}
	waitKey(10);
	while (true) {
		cap >> frame;
		resize(frame, frame, Size(WIDTH, HEIGHT));
		
		string myText = "Please locate your Paper in square (and Press 'C' Key)"; // + in 5s ~ 4s ~ 3s
		putText(frame, myText, Point(10, 40), 2, 1.2, Scalar(0, 0, 255));
		rectangle(frame, Rect(setP, Point(setP.x + SQ + 2, setP.y + SQ + 2)), Scalar(0, 0, 255), 1, 8, 0);
		imshow("Video", frame);
		if (waitKey(60) == 27) {
			return;
		}
		if ((char)waitKey(60) == 'c') {
			pap = frame(bounds & sq).clone();
			break;
		}
	}
	waitKey(10);

	oriR = rock.clone();
	oriS = scis.clone();
	oriP = pap.clone();

	cvtColor(rock, rock, COLOR_BGR2GRAY);
	cvtColor(scis, scis, COLOR_BGR2GRAY);
	cvtColor(pap, pap, COLOR_BGR2GRAY);

	float* rx, * ry, * sx, * sy, * px, * py;
	float* rh, * sh, * ph;

	float* tx, * ty, * th;
	rx = (float*)calloc(SQ * SQ, sizeof(float));
	ry = (float*)calloc(SQ * SQ, sizeof(float));
	sx = (float*)calloc(SQ * SQ, sizeof(float));
	sy = (float*)calloc(SQ * SQ, sizeof(float));
	px = (float*)calloc(SQ * SQ, sizeof(float));
	py = (float*)calloc(SQ * SQ, sizeof(float));

	

	rh = (float*)calloc(9 * HISTO_HEIGHT, sizeof(float));
	sh = (float*)calloc(9 * HISTO_HEIGHT, sizeof(float));
	ph = (float*)calloc(9 * HISTO_HEIGHT, sizeof(float));

	
	Grad(rock, SQ, SQ, rx, ry);
	Grad(scis, SQ, SQ, sx, sy);
	Grad(pap, SQ, SQ, px, py);
	make_histo(rock, SQ, SQ, rx, ry, rh);
	make_histo(scis, SQ, SQ, sx, sy, sh);
	make_histo(pap, SQ, SQ, px, py, ph);
	free(rx);
	free(ry);
	free(sx);
	free(sy);
	free(px);
	free(py);

	float de_r, de_s, de_p;

	imshow("rock", rock);
	imshow("Scissor", scis);
	imshow("Paper", pap);
	waitKey(0);

	init_queue();
	Motion* t;
	Mat clone_frame;
	srand((unsigned)time(NULL));
	while (1) {
		cap >> frame;
		resize(frame, frame, Size(WIDTH, HEIGHT));
		cvtColor(frame, clone_frame, COLOR_BGR2GRAY);
		rectangle(frame, Rect(WIDTH - SQ - SQ / 5, 0, SQ + SQ / 5, HEIGHT), Scalar(0, 255, 0), 3);
		frameCnt++;
		tx = (float*)calloc(SQ * SQ, sizeof(float));
		ty = (float*)calloc(SQ * SQ, sizeof(float));
		th = (float*)calloc(9 * HISTO_HEIGHT, sizeof(float));
		if (frameCnt > 200) {
			frameCnt = 0;
			put(rand() % 3);
		}
		for (t = dhead->next; t != dtail;  t = t->next) {
			if (t->x  >= WIDTH - SQ)
			{
				t->correct_flag = 0;
				get();
			}
			
			else if(t->x >= WIDTH - SQ - SQ/5) {
				temp = clone_frame(bounds & Rect(t->x, t->y - SQ, SQ, SQ)).clone();
				Grad(temp, SQ, SQ, tx, ty);
				make_histo(temp, SQ, SQ, tx, ty, th);
				de_r = de_s = de_p = 0.;
				// ºñ±³ 
				for (int j = 0; j < HISTO_HEIGHT; j++) {
					for (int i = 0; i < 9; i++) {
						de_r += fabs(rh[j * 9 + i] - th[j * 9 + i]);
						de_s += fabs(sh[j * 9 + i] - th[j * 9 + i]);
						de_p += fabs(ph[j * 9 + i] - th[j * 9 + i]);
					}
				}
				if ((de_r + de_s + de_p) * 0.3f > de_r) {
					if (t->key == 1) {
						t->correct_flag = 1; 
						get();
					}
					else {
						rectangle(frame, Rect(t->x, t->y - SQ, SQ, SQ), Scalar(0, 0, 255));
						putText(frame, motion[t->key], Point(t->x, t->y), 2, 1.2, Scalar(0, 0, 255));
					}
				}
				else if ((de_r + de_s + de_p) * 0.3f > de_s) {
					if (t->key == 2) {
						t->correct_flag = 1; 
						get();
					}
					else {
						rectangle(frame, Rect(t->x, t->y - SQ, SQ, SQ), Scalar(0, 0, 255));
						putText(frame, motion[t->key], Point(t->x, t->y), 2, 1.2, Scalar(0, 0, 255));
					}
				}
				else if ((de_r + de_s + de_p) * 0.3f > de_p) {
					if (t->key == 0) {
						t->correct_flag = 1; 
						get();
					}
					else {
						rectangle(frame, Rect(t->x, t->y - SQ, SQ, SQ), Scalar(0, 0, 255));
						putText(frame, motion[t->key], Point(t->x, t->y), 2, 1.2, Scalar(0, 0, 255));
					}
				}
				else {
					printf("rock: %f, scis: %f, pap: %f\n", de_r, de_s, de_p);
					rectangle(frame, Rect(t->x, t->y - SQ, SQ, SQ), Scalar(255, 0, 0));
					putText(frame, motion[t->key], Point(t->x, t->y), 2, 1.2, Scalar(255, 0, 0));
				}
				
			}
			else {
				rectangle(frame, Rect(t->x, t->y - SQ, SQ, SQ), Scalar(0, 0, 255));
				putText(frame, motion[t->key], Point(t->x, t->y), 2, 1.2, Scalar(0, 0, 255));
			}
				
		}
		
		

		
		
		
		system("cls");
		printf("Your score : %d\n", score);
		imshow("Video", frame);
		increase();
		if (waitKey(1) == 27) {
			free(tx);
			free(ty);
			free(th);
			break;
		}
		free(tx);
		free(ty);
		free(th);
	}
	clear_queue();
	free(rh);
	free(sh);
	free(ph);
}
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
void Grad(Mat in, int r, int c, float* X, float* Y) {
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
