#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <time.h>
#include <direct.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#define PI	3.14159265
#define E	2.71828182
#include <math.h>

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

/////////////////////////////////////////////////////////////////////////////

// filter on a single pixel
void _filter(uchar* pixel, const Mat &values, const Mat &kernel, double normValue){
	assert(values.channels() == 1);
	assert(values.size() == kernel.size());
	assert(kernel.type() == CV_64FC1);
	assert(normValue != 0);

	double value = 0;
	for (int y = 0; y < kernel.rows; y++){
		const double *rowKernel = kernel.ptr<double>(y);
		const uchar *rowValues = values.ptr<uchar>(y);
		for (int x = 0; x < kernel.cols; x++){
			value += rowKernel[x] * (double)rowValues[x];
		}
	}
	*pixel = (uchar) max(0., value / normValue);
}

Mat filter(const Mat &img, const Mat &kernel, bool normalize){
	assert(kernel.rows % 2 == 1 && kernel.cols % 2 == 1);
	assert(kernel.type() == CV_64FC1);

	Mat filteredImg(img.rows, img.cols, CV_8UC1, Scalar(0));

	double normValue = 0.;
	for (int y = 0; y < kernel.rows; y++){
		const double *row = kernel.ptr<double>(y);
		for (int x = 0; x < kernel.cols; x++){
			normValue += row[x];
		}
	}

	int yOffset = (kernel.rows - 1) / 2;
	int xOffset = (kernel.cols - 1) / 2;

	for (int y = 0; y < img.rows; y++){
		//copy border from original Image
		if (y < yOffset || y >= img.rows - yOffset){
			img.row(y).copyTo(filteredImg.row(y));
			continue;
		}

		uchar *row = filteredImg.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++){
			//copy border from original Image
			if (x < xOffset || x >= img.cols - xOffset){
				row[x] = img.at<uchar>(y, x);
				continue;
			}

			const Mat tmp = img(Rect(x - xOffset, y - yOffset, kernel.cols, kernel.rows));
			if (normalize)
				_filter(&row[x], tmp, kernel, normValue);
			else
				_filter(&row[x], tmp, kernel, 1.);
		}
	}
	return filteredImg;
}

Mat box(const Mat &img, int kernelHeight, int kernelWidth){
	Mat kernel(kernelHeight, kernelWidth, CV_64FC1, Scalar(1.));
	return filter(img, kernel, true);
}

Mat createGaussianKernel(int kernelHeight, int kernelWidth, double sigma){
	assert(kernelHeight % 2 == 1 && kernelWidth % 2 == 1);

	int yOffset = (kernelHeight - 1) / 2;
	int xOffset = (kernelWidth - 1) / 2;

	Mat kernel(kernelHeight, kernelWidth, CV_64FC1, Scalar(0.));

	for (int y = -yOffset; y <= yOffset; y++){
		double *row = kernel.ptr<double>(y+yOffset);
		for (int x = -xOffset; x <= xOffset; x++){
			row[x+xOffset] = (1. / (2.*PI*sigma*sigma))*pow(E, (-(x*x + y*y) / (2 * sigma*sigma)));
		}
	}
	return kernel;
}

Mat gaussian(const Mat &img, int kernelHeight, int kernelWidth, double sigma){
	Mat kernel = createGaussianKernel(kernelHeight, kernelWidth, sigma);
	return filter(img, kernel, true);
}

// median value of the neighbourhood of a single pixel
void _median(uchar* pixel, const Mat &values, int kernelHeight, int kernelWidth){
	assert(values.channels() == 1);

	vector<uchar> listValues(0);
	for (int y = 0; y < kernelHeight; y++){
		const uchar *rowValues = values.ptr<uchar>(y);
		for (int x = 0; x < kernelWidth; x++){
			listValues.push_back(rowValues[x]);
		}
	}
	assert(listValues.size() % 2 == 1);

	sort(listValues.begin(), listValues.end());
	uchar median = listValues.at((listValues.size() - 1) / 2);

	*pixel = median;
}

Mat median(const Mat &img, int kernelHeight, int kernelWidth){
	assert(kernelHeight % 2 == 1 && kernelWidth % 2 == 1);
	Mat medianImg(img.rows, img.cols, CV_8UC1, Scalar(0));

	int yOffset = (kernelHeight - 1) / 2;
	int xOffset = (kernelWidth - 1) / 2;

	for (int y = 0; y < img.rows; y++){
		//copy border from original Image
		if (y < yOffset || y >= img.rows - yOffset){
			img.row(y).copyTo(medianImg.row(y));
			continue;
		}

		uchar *row = medianImg.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++){
			//copy border from original Image
			if (x < xOffset || x >= img.cols - xOffset){
				row[x] = img.at<uchar>(y, x);
				continue;
			}

			Mat tmp = img(Rect(x - xOffset, y - yOffset, kernelWidth, kernelHeight));
			_median(&row[x], tmp, kernelHeight, kernelWidth);
		}
	}

	return medianImg;
}

// create image from sin frequency
Mat createTestPattern(int w, int h){
	int A = 128, o = 128;

	Mat img(h, w, CV_8UC1, Scalar(0));

	for (int y = 0; y < img.rows; y++){
		uchar* row = img.ptr<uchar>(y);
		for (int x = 0; x < img.cols; x++){
			double xr = x - ((w - 1) / 2.);
			double yr = y - ((h - 1) / 2.);
			double value = (A*sin(0.5*PI*((xr*xr) + (yr*yr)) / h)) + o;
			row[x] = (uchar) value;
		}
	}

	return img;
}

Mat createTestPattern(){
	return createTestPattern(400, 400);
}

/////////////////////////////////////////////////////////////////////////////

int main(){
	int w = 512, h = 512;
	Mat cvTestPattern(h, w, CV_8UC1);
	Mat testPattern = createTestPattern(w, h);
	saveImg("results", "Test Pattern.jpg", testPattern);
	testPattern.copyTo(cvTestPattern);

	for (int size : {3, 5, 7}){
		Mat q0 = testPattern(Rect(0, 0, w / 2, h / 2));
		Mat cvQ0 = cvTestPattern(Rect(0, 0, w / 2, h / 2));
		Mat boxPattern = box(q0, size, size);
		boxPattern.copyTo(q0);
		boxFilter(cvQ0, cvQ0, -1, Size(size, size));

		Mat q1 = testPattern(Rect(w / 2, 0, w / 2, h / 2));
		Mat cvQ1 = cvTestPattern(Rect(w / 2, 0, w / 2, h / 2));
		Mat gaussianPattern = gaussian(q1, size, size, 2.);
		gaussianPattern.copyTo(q1);
		GaussianBlur(cvQ1, cvQ1, Size(size, size), 2.);

		Mat q2 = testPattern(Rect(0, h / 2, w / 2, h / 2));
		Mat cvQ2 = cvTestPattern(Rect(0, h / 2, w / 2, h / 2));
		Mat medianPattern = median(q2, size, size);
		medianPattern.copyTo(q2);
		medianBlur(cvQ2, cvQ2, size);

		saveImg("results", "Filtered Test Pattern_" + to_string(size) + "_" + to_string(size) + ".jpg", testPattern);

		imshow("Test Pattern", testPattern);
		imshow("OpenCV Test Pattern", cvTestPattern);

		waitKey();
		destroyAllWindows();
	}

	return 0;
}