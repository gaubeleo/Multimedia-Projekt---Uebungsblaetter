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
void _filter(short* pixel, const Mat &values, const Mat &kernel, double normValue){
	assert(values.channels() == 1);
	assert(values.size() == kernel.size());
	assert(kernel.type() == CV_64FC1);
	assert(values.type() == CV_8UC1);
	assert(normValue != 0);

	double value = 0.;
	for (int y = 0; y < kernel.rows; y++){
		const double *rowKernel = kernel.ptr<double>(y);
		const uchar *rowValues = values.ptr<uchar>(y);
		for (int x = 0; x < kernel.cols; x++){
			value += rowKernel[x] * (double)rowValues[x];
		}
	}
	*pixel = (short) (value / normValue);
	//cout << normValue << endl;
}

Mat filter(const Mat &img, const Mat &kernel, bool normalize){
	assert(kernel.rows % 2 == 1 && kernel.cols % 2 == 1);
	assert(kernel.type() == CV_64FC1);

	Mat filteredImg(img.rows, img.cols, CV_16SC1, Scalar(0));

	double normValue = 0.;
	for (int y = 0; y < kernel.rows; y++){
		const double *row = kernel.ptr<double>(y);
		for (int x = 0; x < kernel.cols; x++){
			normValue += row[x];
		}
	}

	int yOffset = (kernel.rows - 1) / 2;
	int xOffset = (kernel.cols - 1) / 2;

	for (int y = yOffset; y < img.rows-yOffset; y++){
		short *row = filteredImg.ptr<short>(y);
		for (int x = xOffset; x < img.cols-xOffset; x++){
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
	assert(X.type() == CV_16SC1 && Y.type() == CV_16SC1);

	Mat mag(X.rows, X.cols, CV_16SC1);

	for (int y = 0; y < mag.rows; y++){
		short *row = mag.ptr<short>(y);
		const short *rowX = X.ptr<short>(y);
		const short *rowY = Y.ptr<short>(y);
		for (int x = 0; x < mag.cols; x++){
			// magnitude calculation to match OpenCV, but for displayable Magnitude-Image: don't divide normValue by 4 to achieve same result as OpenCV
			//int value = 0.5*min(255, abs(rowX[x])) + 0.5*min(255, abs(rowY[x]));
			// magnitude calculation using Script
			int value = abs(rowX[x]) + abs(rowY[x]);
			// should not exceed max value of short, but just to be sure:
			row[x] = (short) min(32768, value);
		}
	}
	return mag;
}

double getAbsMax(const Mat &img){
	assert(img.type() == CV_16SC1);

	double normValue = 1.;
	for (int y = 0; y < img.rows; y++){
		const short *row = img.ptr<short>(y);
		for (int x = 0; x < img.cols; x++){
			if (abs(row[x]) > normValue)
				normValue = abs(row[x]);
		}
	}
	assert(normValue != 0);
	return normValue;
}

Mat convertToImg(const Mat &img){
	assert(img.type() == CV_16SC1);

	double normValue = getAbsMax(img);
	// normValue is the absolute maximal value in the original Sobel-Mat
	// however OpenCV seems to be using a fourth of that value to normalize its own Sobel Image Representation
	normValue /= 4;

	Mat convertedImg(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++){
		uchar *row = convertedImg.ptr<uchar>(y);
		const short *rowOrg = img.ptr<short>(y);
		for (int x = 0; x < img.cols; x++){
			row[x] = (uchar) min(255., abs(((double)rowOrg[x]/normValue)*255.));
		}
	}
	return convertedImg;
}

Mat calcGradients(const Mat &X, const Mat &Y){
	assert(X.size() == Y.size());
	assert(X.type() == CV_16SC1 && Y.type() == CV_16SC1);

	Mat gradients(X.rows, X.cols, CV_64FC1);

	for (int y = 0; y < gradients.rows; y++){
		double *row = gradients.ptr<double>(y);
		const short *rowX = X.ptr<short>(y);
		const short *rowY = Y.ptr<short>(y);
		for (int x = 0; x < gradients.cols; x++){
			double direction = atan2(rowY[x], rowX[x]);
			row[x] = direction;
		}
	}
	return gradients;
}

void _drawGradient(Mat &gradImg, int x, int y, double gradDir, short gradMag){
	double length = 0.06;
	int xOffset = (gradMag * cos(gradDir) * length)/2;
	int yOffset = (gradMag * sin(gradDir) * length)/2;

	Point start = Point(x - xOffset, y - yOffset);
	Point end = Point(x + xOffset, y + yOffset);

	line(gradImg, start, end, Scalar(0, 255, 0));
	if (gradDir < 0 && yOffset > 0)
		circle(gradImg, start, 2, Scalar(0, 255, 0), -1);
	else
		circle(gradImg, end, 2, Scalar(0, 255, 0), -1);
	circle(gradImg, Point(x, y), 2, Scalar(0, 0, 255), -1);
}

Mat drawGradients(const Mat &img, const Mat &gradients, const Mat &dervMag){
	assert(img.channels() == 3);
	assert(img.type() == CV_8UC3 && gradients.type() == CV_64FC1 && dervMag.type() == CV_16SC1);
	assert(img.size() == gradients.size() && img.size() == dervMag.size());

	short threshold = 150;

	Mat gradImg = img.clone();
	for (int y = 0; y < img.rows; y+=5){
		const double *rowGrad = gradients.ptr<double>(y);
		const short *rowMag = dervMag.ptr<short>(y);
		for (int x = 0; x < img.cols; x+=5){
			if (rowMag[x] > threshold){
				_drawGradient(gradImg, x, y, rowGrad[x], rowMag[x]);
			}
		}
	}
	return gradImg;
}

/////////////////////////////////////////////////////////////////////////////

int main(){
	Mat img = loadImg("src", "Testimage_gradients.jpg", IMREAD_GRAYSCALE); //IMREAD_COLOR
	//Mat img = loadImg("src", "lenna.jpg", IMREAD_GRAYSCALE); //IMREAD_COLOR

	// Aufgabe a)
	Mat dervX = sobelX(img);
	Mat dervImgX = convertToImg(dervX);
	Mat dervY = sobelY(img);
	Mat dervImgY = convertToImg(dervY);

	// Aufgabe b)
	Mat dervMag = calcMagnitude(dervX, dervY);
	Mat dervImgMag = convertToImg(dervMag);

	// Aufgabe c)
	Mat gradients = calcGradients(dervX, dervY);

	//Aufgabe d)
	Mat colorImg(img.rows, img.cols, CV_8UC3);
	cvtColor(img, colorImg, CV_GRAY2RGB);

	Mat gradImg = drawGradients(colorImg, gradients, dervMag);

	// displaying and saving images

	imshow("Sobel X", dervImgX);
	imshow("Sobel Y", dervImgY);
	imshow("Magnitude", dervImgMag);
	imshow("Gradient Image", gradImg);

	saveImg("results", "Sobel X.jpg", dervImgX);
	saveImg("results", "Sobel Y.jpg", dervImgY);
	saveImg("results", "Magnitude.jpg", dervImgMag);
	saveImg("results", "Gradient Image.jpg", gradImg);

	//// for OpenCV comparison:
	//Mat grad_x, grad_y, Xcv, Ycv, Magcv;

	//Sobel(img, grad_x, CV_16SC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, Xcv);

	//Sobel(img, grad_y, CV_16SC1, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	//convertScaleAbs(grad_y, Ycv);

	//addWeighted(Xcv, 0.5, Ycv, 0.5, 0, Magcv);

	//imshow("Sobel X CV", Xcv);
	//imshow("Sobel Y CV", Ycv);
	//imshow("Magnitude CV", Magcv);
	//// end OpenCV

	waitKey();
	destroyAllWindows();

	return 0;
}