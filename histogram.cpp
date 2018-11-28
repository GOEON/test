// https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html

#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("lena.jpg");
	
	// b, g, r 채널 분리 
	vector<Mat> bgr_planes;
	split(img, bgr_planes);
	
	// bin 몇개로 할지 
	int histSize = 256;

	// range 
	float range[] = { 0,256 }; // upper boundary is exclusive 
	const float * histRange = { range };

	bool uniform = true;
	bool accumulate = false; 

	Mat b_hist, g_hist, r_hist;

	// B, G, R 채널 각각에 대해 1차원 히스토그램 계산 
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// histogram 그릴 윈도우 
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double) hist_w / histSize);

	Mat histImageB(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageG(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImageR(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	normalize(b_hist, b_hist, 0, histImageB.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImageG.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImageR.rows, NORM_MINMAX, -1, Mat());

	/*for (int i = 1; i < histSize; i++) {
		line(histImageB, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 1, 8, 0);
		line(histImageG, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 1, 8, 0);
		line(histImageR, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 1, 8, 0);
	}*/

	for (int i = 1; i < histSize; i++) {
		rectangle(histImageB, Point(bin_w * (i - 1), hist_h),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 1, 8, 0);
		rectangle(histImageG, Point(bin_w * (i - 1), hist_h),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 1, 8, 0);
		rectangle(histImageR, Point(bin_w * (i - 1), hist_h),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 1, 8, 0);
	}


	imshow("original image", img);
	imshow("blue histogram", histImageB);
	imshow("green histogram", histImageG);
	imshow("red histogram", histImageR);
		
	waitKey();
	return 0;
}