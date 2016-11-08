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

	for (int y = 0; y < img.rows; y++){
		//copy border from original Image
		if (y < yOffset || y >= img.rows - yOffset){
			img.row(y).copyTo(filteredImg.row(y));
			continue;
		}

		short *row = filteredImg.ptr<short>(y);
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
	assert(X.type() == CV_16SC1 && Y.type() == CV_16SC1);

	Mat mag(X.rows, X.cols, CV_8UC1);

	for (int y = 0; y < mag.rows; y++){
		uchar *row = mag.ptr<uchar>(y);
		const short *rowX = X.ptr<short>(y);
		const short *rowY = Y.ptr<short>(y);
		for (int x = 0; x < mag.cols; x++){
			int value = 0.5*abs(rowX[x]) + 0.5*abs(rowY[x]);
			row[x] = (uchar) min(255, value);
		}
	}
	return mag;
}

Mat convertToImg(const Mat &img){
	assert(img.type() == CV_16SC1);

	double normValue = 1.;
	for (int y = 0; y < img.rows; y++){
		const short *row = img.ptr<short>(y);
		for (int x = 0; x < img.cols; x++){
			if (abs(row[x]) > normValue)
				normValue = abs(row[x]);
		}
	}

	Mat convertedImg(img.rows, img.cols, CV_8UC1);
	for (int y = 0; y < img.rows; y++){
		uchar *row = convertedImg.ptr<uchar>(y);
		const short *rowOrg = img.ptr<short>(y);
		for (int x = 0; x < img.cols; x++){
			row[x] = (uchar) abs(((double)rowOrg[x]/normValue)*255.);
		}
	}
	return convertedImg;
}

//Mat calcGradientDirection(const Mat &mag){
//	//
//}

/////////////////////////////////////////////////////////////////////////////

int main(){
	Mat img = loadImg("src", "Testimage_gradients.jpg", IMREAD_GRAYSCALE); //IMREAD_COLOR

	// Aufgabe a)
	Mat dervX = sobelX(img);
	Mat dervImgX = convertToImg(dervX);
	Mat dervY = sobelY(img);
	Mat dervImgY = convertToImg(dervY);

	// Aufgabe b)
	Mat dervMag = calcMagnitude(dervX, dervY);

	// Aufgabe c)
	//Mat gradientDir = calcGradientDirection(mag);

	Mat grad_x, grad_y, Xcv, Ycv, Magcv;

	Sobel(img, grad_x, CV_16SC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, Xcv);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(img, grad_y, CV_16SC1, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, Ycv);

	/// Total Gradient (approximate)
	addWeighted(Xcv, 0.5, Ycv, 0.5, 0, Magcv);

	imshow("Sobel X", dervImgX);
	imshow("Sobel Y", dervImgY);
	imshow("Magnitude", dervMag);

	imshow("Sobel X CV", Xcv);
	imshow("Sobel Y CV", Ycv);
	imshow("Magnitude CV", Magcv);

	waitKey();
	destroyAllWindows();

	return 0;
}