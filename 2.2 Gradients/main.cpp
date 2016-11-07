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
/////////////////////////////////////////////////////////////////////////////

Mat sobelX(const Mat &img){
	Mat kernel((Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1));
	return filter(img, kernel, false);
}

Mat sobelY(const Mat &img){
	Mat kernel((Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1));
	return filter(img, kernel, false);
}

Mat calcMagnitude(const Mat &X, const Mat &Y){
	assert(X.size() == Y.size());
	assert(X.type() == CV_8UC1 && Y.type() == CV_8UC1);

	Mat mag(X.rows, X.cols, CV_8UC1);

	for (int y = 0; y < mag.rows; y++){
		uchar *row = mag.ptr<uchar>(y);
		const uchar *rowX = X.ptr<uchar>(y);
		const uchar *rowY = Y.ptr<uchar>(y);
		for (int x = 0; x < mag.cols; x++){
			int value = (int)rowX[x] + (int)rowY[x];
			row[x] = (uchar)min(255, value);
		}
	}
	return mag;
}

Mat calcGradientDirection(const Mat &mag){
	//
}

/////////////////////////////////////////////////////////////////////////////

int main(){
	Mat img = loadImg("src", "lenna.jpg", IMREAD_GRAYSCALE); //IMREAD_COLOR

	// Aufgabe a)
	Mat X = sobelX(img);
	Mat Y = sobelY(img);

	// Aufgabe b)
	Mat mag = calcMagnitude(X, Y);

	// Aufgabe c)
	Mat gradientDir = calcGradientDirection(mag);

	imshow("Sobel X", X);
	imshow("Sobel Y", Y);
	imshow("Magnitude", mag);

	waitKey();
	destroyAllWindows();

	return 0;
}