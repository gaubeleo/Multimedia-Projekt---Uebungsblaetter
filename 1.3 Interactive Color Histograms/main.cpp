#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <time.h>
#include <direct.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;


Mat loadImg(string directory, string filename, int flags){
	string fullFilename = string(directory + "\\" + filename);
	Mat image;
	image = imread(fullFilename, flags);

	if (!image.data){
		cout << "image file " << fullFilename << " could not be opened" << endl;
		getchar();
		exit(-1);
	}

	return image;
}

bool saveImg(string directory, string filename, Mat img){
	string fullFilename = string(directory + "\\" + filename);
	struct stat sb;

	if (!(stat(directory.c_str(), &sb) == 0 && sb.st_mode == S_IFDIR)){
		_mkdir(directory.c_str());
	}

	cout << "successfully written '" << fullFilename << "' to file!" << endl;

	return imwrite(fullFilename, img);
}

vector<Mat> splitChannels(Mat img) {
	assert(img.channels() == 3);

	vector<Mat> channels;

	//split(img, channels);

	for (int i = 0; i < 3; i++) {
		channels.push_back(Mat(img.rows, img.cols, CV_8UC1));
	}

	for (int y = 0; y < img.rows; y++) {
		Vec3b* row = img.ptr<Vec3b>(y);
		for (int x = 0; x < img.cols; x++) {
			for (int c = 0; c < 3; c++) {
				channels[c].at<uchar>(Point(x, y)) = row[x][c];
			}
		}
	}

	return channels;
}

uchar quantize(uchar value, unsigned binCount){
	return (unsigned)((double)value / 256. * binCount);
}

vector<int> calcHistogram(Mat img, unsigned binCount){
	assert(img.channels() == 1);
	//assert(binCount < (1 << img.depth()));

	vector<int> histogramValues(binCount, 0);

	for (int y = 0; y < img.rows; y++){
		uchar* row = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++){
			uchar value = row[x];
			unsigned bin = quantize(value, binCount);
			histogramValues[bin]++;
		}
	}
	return histogramValues;
}

// normalizes using a fixed value
Mat createHistogramImage(vector<int> histogramValues, int normalizeValue) {
	int binCount = (int)histogramValues.size();
	int width = 256 * 2 / binCount*binCount;
	int height = 400;
	int binWidth = width / binCount;

	Mat histogram(height, width, CV_8UC1, Scalar(255));

	double magnitude;
	for (int b = 0; b < binCount; b++) {
		magnitude = min(1., (double)histogramValues[b] / (double)normalizeValue);
		// draws #bin_width lines next to each over
		for (int j = 0; j < binWidth; j++) {
			line(histogram, Point((b*binWidth) + j, height - 1), Point((b*binWidth) + j, height - (int)(magnitude*height)), Scalar(0));
		}
	}
	return histogram;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat cropImg(Mat img, Rect rect){
	return img(Range(rect.y, rect.y + rect.height), Range(rect.x, rect.x + rect.width)).clone();
}

Rect drawRect(Mat img, unsigned x1, unsigned y1, unsigned x0, unsigned y0, Scalar color){
	unsigned x = min(x0, x1);
	unsigned y = min(y0, y1);
	unsigned width = max(x0, x1)-x;
	unsigned height = max(y0, y1)-y;

	Rect rect(x, y, width, height);
	rectangle(img, rect, color, 2);

	imshow("Image", img);

	return rect;
}

void cropRectImg(Mat img, Rect rect){
	Mat croppedImg = cropImg(img, rect);
	vector<Mat> croppedImgChannels = splitChannels(croppedImg);

	Mat histogramRed = createHistogramImage(calcHistogram(croppedImgChannels[2], 100), 10000);
	Mat histogramGreen = createHistogramImage(calcHistogram(croppedImgChannels[1], 100), 10000);
	Mat histogramBlue = createHistogramImage(calcHistogram(croppedImgChannels[0], 100), 10000);

	imshow("Cropped Image", croppedImg);

	imshow("Cropped Histogram Red", histogramRed);
	imshow("Cropped Histogram Green ", histogramGreen);
	imshow("Cropped Histogram Blue", histogramBlue);

	cout << "Rect at - " << "x: " << rect.x << ", y: " << rect.y << ", width: " << rect.width << ", height: " << rect.height << endl;
}

struct Mouse{
	bool leftDown = false;
	unsigned xClick = 0;
	unsigned yClick = 0;
} MOUSE;

void callBackFunc(int event, int x, int y, int flags, void* userdata)
{
	Mat img = *(Mat*)userdata;
	if (event == EVENT_LBUTTONDOWN)
	{
		unsigned x0 = min(max(0, x), img.cols);
		unsigned y0 = min(max(0, y), img.rows);
		
		MOUSE.xClick = x;
		MOUSE.yClick = y;

		MOUSE.leftDown = true;
	}
	else if (event == EVENT_LBUTTONUP && MOUSE.leftDown)
	{
		unsigned x1 = min(max(0, x), img.cols);
		unsigned y1 = min(max(0, y), img.rows);

		if (x1 != MOUSE.xClick && y1 != MOUSE.yClick){
			Rect rect = drawRect(img.clone(), x1, y1, MOUSE.xClick, MOUSE.yClick, Scalar(0, 255, 0));
			cropRectImg(img, rect);
		}
		
		MOUSE.leftDown = false;
	}
	else if (event == EVENT_MOUSEMOVE && MOUSE.leftDown)
	{
		unsigned x1 = min(max(0, x), img.cols);
		unsigned y1 = min(max(0, y), img.rows);

		drawRect(img.clone(), x1, y1, MOUSE.xClick, MOUSE.yClick, Scalar(0, 0, 255));
	}
}

void waitForMouseDrag(Mat img){
	namedWindow("Image");

	setMouseCallback("Image", callBackFunc, &img);
	imshow("Image", img);

	waitKey();
	destroyAllWindows();
}


int main(){
	Mat img = loadImg("src", "IMG_6211.jpg", IMREAD_COLOR);

	// Aufgabe a) + b)
	waitForMouseDrag(img);

	return 0;
}