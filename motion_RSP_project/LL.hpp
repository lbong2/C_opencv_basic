#pragma once
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>
#include <random>
using namespace cv;
using namespace std;
#define SQ 256
#define WIDTH 1280
#define HEIGHT 960
#define HISTO_HEIGHT (SQ / 8 - 1) * (SQ / 8 - 1)
typedef struct _dnode
{
	int key; // 0~2
	int x, y;
	int correct_flag = 0;
	struct _dnode* prev;
	struct _dnode* next;
}Motion;
int frameCnt = 0;
int score = 0;
Motion* dhead, * dtail;
random_device rd;
void init_queue()
{
	
	dhead = (Motion*)calloc(1, sizeof(Motion));
	dtail = (Motion*)calloc(1, sizeof(Motion));
	dhead->prev = dhead;
	dhead->next = dtail;
	dtail->prev = dhead;
	dtail->next = dtail;
}
int put(int k)
{
	Motion* t;
	if ((t = (Motion*)malloc
	(sizeof(Motion))) == NULL)
	{
		printf("Out of memory !\n");
		return -1;
	}
	t->key = k;
	
	mt19937 gen(rd());
	uniform_int_distribution<int> dis(256, HEIGHT - 1);
	t->x = 0;
	t->y = dis(gen);
	dtail->prev->next = t;
	t->prev = dtail->prev;
	dtail->prev = t;
	t->next = dtail;
	frameCnt++;
	return k;
}
int get()
{
	Motion* t;
	int k;
	t = dhead->next;
	if (t == dtail)
	{
		printf("Queue underflow\n");
		return -1;
	}
	k = t->key;
	dhead->next = t->next;
	t->next->prev = dhead;
	if (t->correct_flag == 1)
		score++;
	else
		score--;
	free(t);
	frameCnt--;
	return k;
}
void clear_queue()
{
	Motion* t, * s;
	t = dhead->next;
	while (t != dtail)
	{
		s = t;
		t = t->next;
		free(s);
	}
	dhead->next = dtail;
	dtail->prev = dhead;
}
void increase()
{
	Motion* t;
	t = dhead->next;
	while (t != dtail) {
		(t->x) += 3;
		t = t->next;
	}
}
void print_queue()
{
	Motion* t;
	t = dhead->next;
	while (t != dtail) {
		printf("%s", t->key);
		t = t->next;
	}
}