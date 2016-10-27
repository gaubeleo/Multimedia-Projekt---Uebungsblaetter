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

void displaySeperateChannels(Mat img){
	string name;
	vector<Mat> channels = splitChannels(img);

	// Reverse because OPEN_CV uses BGR instead of RGB
	imshow("RED", channels[2]);
	imshow("GREEN", channels[1]);
	imshow("BLUE", channels[0]);

	waitKey();
	destroyAllWindows();

	saveImg("results", "RED.jpg", channels[2]);
	saveImg("results", "GREEN.jpg", channels[1]);
	saveImg("results", "BLUE.jpg", channels[0]);
}

unsigned quantize(uchar value, unsigned binCount){
	return (unsigned)((double)value / 256. * binCount);
}

void calc3DHistogram(Mat img, unsigned binCount){
	assert(img.channels() == 3);
	assert(binCount <= 1 << img.depth());

	// 3D-histogram in BGR-format
	vector<vector<vector<int>>> histogram(256, vector<vector<int>>(256, vector<int>(256)));

	for (int y = 0; y < img.rows; y++){
		Vec3b* row = img.ptr<Vec3b>(y);
		for (int x = 0; x < img.cols; x++){
			Vec3b pixel = row[x];
			histogram[quantize(pixel[0], binCount)][quantize(pixel[1], binCount)][quantize(pixel[2], binCount)]++;
		}
	}

	unsigned emptyVals = 0;
	for (unsigned b = 0; b < binCount;b++){
		for (unsigned g = 0; g < binCount; g++){
			for (unsigned r = 0; r < binCount; r++){
				histogram[b][g][r] == 0 ? emptyVals++ : emptyVals;
			}
		}
	}

	cout << "ratio of empty / total bins for histogram with "<< binCount << " bins: " << (double)emptyVals/(binCount*binCount*binCount) << endl;
}

void reverseChannels(Mat img){
	Mat reversedImg = Mat(img.rows, img.cols, CV_8UC3);

	for (int y = 0; y < img.rows; y++){
		Vec3b* row = img.ptr<Vec3b>(y);
		Vec3b* rowReversed = reversedImg.ptr<Vec3b>(y);
		for (int x = 0; x < img.cols; x++){
			Vec3b pixel = row[x];
			rowReversed[x] = Vec3b(pixel[2], pixel[1], pixel[0]);
		}
	}
	imshow("Reversed Channels", reversedImg);

	waitKey();
	destroyAllWindows();

	saveImg("results", "BGR_img.jpg", reversedImg);
}

Vec3b BGR2HSV(Vec3b bgrColor){
	// Converting a single Color from BGR to HSV
	Mat3b _bgrColor(bgrColor);
	Mat3b _hsvColor;
	cvtColor(_bgrColor, _hsvColor, COLOR_BGR2HSV);

	return Vec3b(_hsvColor.at<uchar>(0), _hsvColor.at<uchar>(1), _hsvColor.at<uchar>(2));
}

uchar calcGreayscale(Vec3b pixel){
	return (uchar)(0.2126*pixel[2] + 0.7152*pixel[1] + 0.0722*pixel[2]);
}

bool insideRange(uchar x, uchar min, uchar max){
	return x >= min ? x <= max : false;
}

bool inHSVRange(Vec3b hsvPixel, Vec3b hsvColor, uchar radius){
	uchar h = hsvPixel[0];
	uchar s = hsvPixel[1];
	uchar v = hsvPixel[2];
	if (insideRange(h, max(hsvColor[0] - radius, 0), min(hsvColor[0] + radius, 179))
		&& insideRange(s, 50, 255)
		&& insideRange(v, 50, 255)) {
		return true;
	}
	return false;
}

void highlightHue(Mat img, Vec3b bgrColor, uchar radius){
	Mat hsvImg, highlightedImg(img.rows, img.cols, CV_8UC3);
	cvtColor(img, hsvImg, COLOR_BGR2HSV);

	Vec3b hsvColor = BGR2HSV(bgrColor);

	for (int y = 0; y < img.rows; y++){
		Vec3b* row = img.ptr<Vec3b>(y);
		Vec3b* rowHSV = hsvImg.ptr<Vec3b>(y);
		Vec3b* rowHighlighted = highlightedImg.ptr<Vec3b>(y);
		for (int x = 0; x < img.cols; x++){
			Vec3b pixelHSV = rowHSV[x];
			if (inHSVRange(pixelHSV, hsvColor, radius)){
				rowHighlighted[x] = row[x];
			}
			else{
				uchar greyValue = calcGreayscale(row[x]);
				rowHighlighted[x] = Vec3b(greyValue, greyValue, greyValue);
			}
		}
	}

	imshow("Highlight Hue", highlightedImg);

	waitKey();
	destroyAllWindows();

	saveImg("results", "Highlighted_Hue.jpg", highlightedImg);
}

int main(){
	Mat img = loadImg("src", "DSC_0078.jpg", IMREAD_COLOR);
	Mat testImg = loadImg("src", "test_image.jpg", IMREAD_COLOR);

	//Aufgabe a)
	displaySeperateChannels(img);

	// Aufgabe b)
	for (int n : {1, 2, 4, 6, 7, 8}){
		calc3DHistogram(img, 1 << n);
	}

	//Aufgabe c)
	reverseChannels(img);

	//Aufgabe d)
	highlightHue(testImg, Vec3b(255, 0, 0), 10);

	return 0;
}