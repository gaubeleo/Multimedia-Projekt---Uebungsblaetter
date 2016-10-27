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

void drawRect(Mat img, unsigned x1, unsigned y1, unsigned x0, unsigned y0){
	unsigned x = min(x0, x1);
	unsigned y = min(y0, y1);
	unsigned width = max(x0, x1)-x;
	unsigned height = max(y0, y1)-y;

	rectangle(img, Rect(x, y, width, height), Scalar(0, 0, 255), 2);

	cout << "Rect at - " << "x: " << x << ", y: " << y << ", width: " << width << ", height: " << height << endl;

	imshow("Image", img);
}

struct Mouse{
	bool leftDown = false;
	unsigned xClick = 0;
	unsigned yClick = 0;
} MOUSE;

void callBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		MOUSE.leftDown = true;
		MOUSE.xClick = x;
		MOUSE.yClick = y;
	}
	else if (event == EVENT_LBUTTONUP && MOUSE.leftDown)
	{
		Mat img = *(Mat*)userdata;
		drawRect(img.clone(), x, y, MOUSE.xClick, MOUSE.yClick);
		MOUSE.leftDown = false;

	}
	//else if (event == EVENT_MOUSEMOVE)
	//{
	//	if (MOUSE.leftDown)
	//		cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
	//}
}

uchar quantize(uchar value, unsigned binCount){
	return (unsigned)((double)value / 256. * binCount);
}

vector<int> calcHistogram(Mat img, unsigned binCount){
	assert(img.channels() == 1);
	assert(binCount < (1 << img.depth()));

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
	int width = 256 * 3 / binCount*binCount;
	int height = 500;
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

void waitForMouseDrag(Mat img){
	namedWindow("Image");
	namedWindow("Histogram");

	Mat histogram = createHistogramImage(calcHistogram(img, 300), 20000);

	setMouseCallback("Image", callBackFunc, &img);
	imshow("Image", img);
	imshow("Histogram", histogram);

	waitKey();
	destroyAllWindows();
}


int main(){
	Mat img = loadImg("src", "IMG_6211.jpg", IMREAD_COLOR);

	// Aufgabe a)
	waitForMouseDrag(img);

	return 0;
}