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

uchar quantize(uchar value, unsigned binCount){
	return (unsigned)((double)value / 256. * binCount);
}

vector<int> calcHistogram(Mat img, unsigned binCount){
	assert(img.channels() == 1);
	//assert(binCount <= (1<<img.depth()));

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
	int width = 256*3 / binCount*binCount;
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

// automatically normalizes using the max value of the histogram
Mat createHistogramImage(vector<int> histogramValues) {
	int min = 0;
	int max = 0;

	for (int value : histogramValues) {
		if (value < min) {
			min = value;
		}
		if (value > max) {
			max = value;
		}
	}

	return createHistogramImage(histogramValues, max);
}

Mat enhanceContrast(Mat img, double cutOff){
	assert(img.channels() == 1);

	vector<int> histogramValues = calcHistogram(img, 256);

	int totalCount = 0;
	for (int value : histogramValues){
		totalCount += value;
	}
	int lowBound = 0;
	int highBound = 0;
	int currentCount = 0;
	for (int b = 0; b < 256; b++){
		currentCount += histogramValues[b];
		if (lowBound == 0 && currentCount > cutOff*totalCount){
			lowBound = b;
		}
		else if (currentCount > (1.0 - cutOff)*totalCount){
			highBound = b;
			break;
		}
	}

	Mat enhancedImg(img.rows, img.cols, CV_8UC1, Scalar(0));

	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			uchar value = img.at<uchar>(y, x);
			uchar newValue = min(255, max(0, (int)((double)(value - lowBound) / (double)(highBound - lowBound) * 255)));
			enhancedImg.at<uchar>(y, x) = newValue;
		}
	}

	return enhancedImg;
}

void display(string name, Mat img, Mat histogram){
	imshow(name, img);
	imshow("Histogram", histogram);

	waitKey();
	destroyAllWindows();
}

int main(){
	Mat lenna = loadImg("src", "lenna.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat overflowImg = loadImg("src", "overflow.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat underflowImg = loadImg("src", "underflow.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//Aufgabe a)
	vector<int> lennaHistogram = calcHistogram(lenna, 75);
	Mat lennaHistogramImage = createHistogramImage(lennaHistogram);
	display("Lenna", lenna, lennaHistogramImage);

	//Aufgabe b)
	Mat overflowHistogramImage, underflowHistogramImage;
	for (int b : {64, 128, 256}) {
		overflowHistogramImage = createHistogramImage(calcHistogram(overflowImg, b), 20000);

		display("Overflow Image", overflowImg, overflowHistogramImage);
		saveImg("results", "OverflowHistogram_" + to_string(b) + "_bins.jpg", overflowHistogramImage);
	}
	for (int b : {64, 128, 256}) {
		underflowHistogramImage = createHistogramImage(calcHistogram(underflowImg, b), 20000);

		display("Underflow Image", underflowImg, underflowHistogramImage);
		saveImg("results", "UnderflowHistogram_" + to_string(b) + "_bins.jpg", underflowHistogramImage);
	}

	//Aufgabe c)

	Mat enhancedOverflow = enhanceContrast(overflowImg, 0.05);
	Mat enhancedUnderflow = enhanceContrast(underflowImg, 0.05);

	Mat enhancedOverflowHistogramImage = createHistogramImage(calcHistogram(enhancedOverflow, 256), 20000);
	Mat enhancedUnderflowHistogramImage = createHistogramImage(calcHistogram(enhancedUnderflow, 256), 20000);

	display("Enhanced Overflow Image", enhancedOverflow, enhancedOverflowHistogramImage);
	display("Enhanced Underflow Image", enhancedUnderflow, enhancedUnderflowHistogramImage);
	
	saveImg("results", "enhancedOverflow.jpg", enhancedOverflow);
	saveImg("results", "enhancedOverflowHistogram.jpg", enhancedOverflowHistogramImage);

	saveImg("results", "enhancedUnderflow.jpg", enhancedUnderflow);
	saveImg("results", "enhancedUnderflowHistogram.jpg", enhancedUnderflowHistogramImage);

	return 0;
}