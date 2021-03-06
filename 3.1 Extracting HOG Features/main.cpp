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

#define PI	3.14159265

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
	*pixel = (short)(value / normValue);
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

	for (int y = yOffset; y < img.rows - yOffset; y++){
		short *row = filteredImg.ptr<short>(y);
		for (int x = xOffset; x < img.cols - xOffset; x++){
			const Mat tmp = img(Rect(x - xOffset, y - yOffset, kernel.cols, kernel.rows));
			if (normalize)
				_filter(&row[x], tmp, kernel, normValue);
			else
				_filter(&row[x], tmp, kernel, 1.);
		}
	}
	return filteredImg;
}

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
			row[x] = (short)min(32768, value);
		}
	}
	return mag;
}

// gradients orentation interval ]0, pi[
Mat calcGradients(const Mat &X, const Mat &Y){
	assert(X.size() == Y.size());
	assert(X.type() == CV_16SC1 && Y.type() == CV_16SC1);

	Mat gradients(X.rows, X.cols, CV_64FC1);

	for (int y = 0; y < gradients.rows; y++){
		double *row = gradients.ptr<double>(y);
		const short *rowX = X.ptr<short>(y);
		const short *rowY = Y.ptr<short>(y);
		for (int x = 0; x < gradients.cols; x++){
			double direction = abs(atan2(rowY[x], rowX[x]));
			row[x] = direction;
		}
	}
	return gradients;
}

// gradients orentation interval ]0, pi[
Mat calcGradients(Mat img){
	const Mat X = sobelX(img);
	const Mat Y = sobelY(img);

	return calcGradients(X, Y);
}

/////////////////////////////////////////////////////////////////////////////



double toDegree(double radiant){
	assert(radiant >= 0 && radiant <= 2*PI);

	double degree = (radiant * 360.) / (2 * PI);

	return degree;
}

double toRadiant(double degree){
	assert(degree >= 0 && degree <= 360.);

	double radiant = (degree * 2 * PI) / (360.);

	return radiant;
}



// bin0 --> 10�
// bin1 --> 30�
// bin2 --> 50�
// bin3 --> 70�
// bin4 --> 90�
// bin5 --> 110�
// bin6 --> 130�
// bin7 --> 150�
// bin8 --> 170�

// 0� --> each bin0 and bin8 get 0.5*magnitude

double*** compute_HoG(const Mat &img, const int cellSize, const std::vector<int> &dims){
	assert(img.type() == CV_8UC1);
	assert(dims.size() == 3);

	const Mat X = sobelX(img);
	const Mat Y = sobelY(img);

	Mat gradients = calcGradients(X, Y);
	Mat magnitude = calcMagnitude(X, Y);

	assert(gradients.type() == CV_64FC1);
	assert(magnitude.type() == CV_16SC1);

	const int cellRows = dims.at(0);
	const int cellCols = dims.at(1);
	const int binCount = dims.at(2);

	double*** HoG = (double***)malloc(sizeof(double**)* cellRows);
	if (HoG == NULL)
		exit(1);

	for (int yCell = 0; yCell < cellRows; yCell++){
		HoG[yCell] = (double**)malloc(sizeof(double*)* cellCols);
		if (HoG[yCell] == NULL)
			exit(1);
		for (int xCell = 0; xCell< cellCols; xCell++){
			HoG[yCell][xCell] = (double*)malloc(sizeof(double)* binCount);
			if (HoG[yCell][xCell] == NULL)
				exit(1);
			for (int b = 0; b < binCount; b++)
				HoG[yCell][xCell][b] = 0.;
		}
	}

	double orientation, degree;
	for (int y = 0; y < img.rows - (img.rows%cellSize); y++){
		const double* gradRow = gradients.ptr<double>(y);
		const short* magRow = magnitude.ptr<short>(y);
		int yCell = y / cellSize;
		for (int x = 0; x < img.cols - (img.cols%cellSize); x++){
			if (magRow[x] == 0)
				continue;

			int xCell = x / cellSize;
			orientation = gradRow[x];
			degree = toDegree(orientation);

			double bin = (degree - 0.5*(180. / binCount)) / (180. / binCount);
			if (bin < 0)
				bin += binCount;

			int lowerBin = (int)bin;
			int upperBin = (lowerBin == binCount - 1) ? 0 : lowerBin + 1;

			double lowerBinValue = (1. - (bin - (int)bin));
			double upperBinValue = 1. - lowerBinValue;

			//cout << yCell << ", " << xCell << ", " << lowerBin << ", " << upperBin << ", " << endl;

			HoG[yCell][xCell][lowerBin] += lowerBinValue; // * magRow[x];
			HoG[yCell][xCell][upperBin] += upperBinValue; // * magRow[x];
		}
	}
	return HoG;
}

Mat visualizeHoG(double*** HoG, const int cellSize, const std::vector<int> &dims){
	assert(dims.size() == 3);

	const uchar tau = 30;

	const int cellRows = dims.at(0);
	const int cellCols = dims.at(1);
	const int binCount = dims.at(2);

	int height = cellRows*cellSize;
	int width = cellCols*cellSize;

	Mat HoGimage(height, width, CV_8UC1, Scalar(0));

	for (int yCell = 0; yCell < cellRows; yCell++){
		for (int xCell = 0; xCell< cellCols; xCell++){
			double max = -1, min = -1;
			for (int b = 0; b < binCount; b++){
				double value = HoG[yCell][xCell][b];
				if (value == 0)
					continue;
				if (max == -1 || value > max)
					max = value;
				if (min == -1 || value < min)
					min = value;
			}
			if (min == -1 || max == -1){
				//cout << "empty gradient" << endl;
				continue;
			}

			for (int b = 0; b < binCount; b++){
				double HoGvalue = HoG[yCell][xCell][b];
				if (HoGvalue == 0)
					continue;

				int degree = ((b * 180) / binCount) + (int)(0.5*(180. / binCount)); // bin0 --> 10�, ..., bin8 --> 170�
				double gradDir = toRadiant(degree);

				int centerX = xCell*cellSize + cellSize/2;
				int centerY = yCell*cellSize + cellSize/2;

				int length = cellSize;

				int xOffset = (int)((cos(gradDir) * length)/2);
				int yOffset = (int)((sin(gradDir) * length)/2);

				Point start(centerX+xOffset, centerY+yOffset);
				Point end(centerX-xOffset, centerY-yOffset);

				//uchar strength = 255 * (HoGvalue / max);
				uchar strength = tau + (uchar)((255-tau) * ((HoGvalue-min)/(max-min)));

				line(HoGimage, start, end, Scalar(strength));
			}
		}
	}


	return HoGimage;
}

/////////////////////////////////////////////////////////////////////////////

int main(){
	//Mat img = loadImg("src", "Testimage_gradients.jpg", IMREAD_GRAYSCALE); //IMREAD_COLOR
	//Mat img = loadImg("src", "lenna.jpg", IMREAD_GRAYSCALE);
	Mat img = loadImg("src", "eye.png", IMREAD_GRAYSCALE);

	// Aufgabe 3.1 a)
	Mat gradients = calcGradients(img);

	// Augabe 3.1 b) + 3.2 a) + b)
	for (int cellSize : {10, 20, 30}){
		const int binCount = 9;
		const int cellRows = img.rows / cellSize;
		const int cellCols = img.cols / cellSize;

		vector<int> dims = { cellRows, cellCols, binCount };

		// Aufgabe 3.1 b)
		double*** HoG = compute_HoG(img, cellSize, dims);

		// Aufgabe 3.2 a) + b) 
		Mat HoGimage = visualizeHoG(HoG, cellSize, dims);

		saveImg("results", "HoG_Visualization_" + to_string(cellSize) + ".jpg", HoGimage);

		imshow("HoG Visualization", HoGimage);
		waitKey();
		destroyAllWindows();
	}

	return 0;
}