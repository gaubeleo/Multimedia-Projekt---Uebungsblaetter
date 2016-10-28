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

Mat simulateLowRes(Mat img, int n){
	assert(n > 0);
	assert(img.channels() == 1);

	//use float Mat to save values properly
	Mat lowResImg(img.rows - (img.rows%n), img.cols - (img.cols%n), CV_32FC1, Scalar(0.));

	for (int y = 0; y < lowResImg.rows; y++){
		uchar* row = img.ptr<uchar>(y);
		for (int x = 0; x < lowResImg.cols; x++){
			uchar value = row[x];
			for (int yOff = 0; yOff < n; yOff++){
				for (int xOff = 0; xOff < n; xOff++){
					lowResImg.at<float>((y / n)*n + yOff, (x / n)*n + xOff) += ((float)value) / (n*n);
				}
			}
		}
	}
	//Convert back to unsigned char Mat
	lowResImg.convertTo(lowResImg, CV_8UC1);

	saveImg("results", "lowResImg_" + to_string(n) + ".jpg", lowResImg);

	imshow("low resolution image", lowResImg);
	waitKey();
	destroyAllWindows();

	return lowResImg;
}

Mat quantizeImg(Mat img, int q){
	assert(img.channels() == 1);

	Mat quantizedImg(img.rows, img.cols, CV_8UC1);

	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			uchar value = img.at<uchar>(y, x);
			uchar newValue = ((value >> (8 - q)) << (8 - q)) + 256 / (1 << q + 1);
			quantizedImg.at<uchar>(y, x) = newValue;
		}
	}

	saveImg("results", "quantizedImg_" + to_string(q) + ".jpg", quantizedImg);

	imshow("low resolution image", quantizedImg);
	waitKey();
	destroyAllWindows();

	return quantizedImg;
}

int main(int argc, char* argv[]) {
	if (argc != 4){
		cout << "this program needs to be run with three command line parameters: <filename> <int> <int>" << endl;
		cout << "example: '1.4 Pixel Manipulation.exe' \"src\\lenna.jpg\" 3 3" << endl;
		return -1;
	}

	string dir = "";
	string filename = "";

	string fullFilename(argv[1]);

	dir = fullFilename.substr(0, fullFilename.rfind("\\"));
	filename = fullFilename.substr(fullFilename.rfind("\\")+1);

	Mat img = loadImg(dir, filename, IMREAD_GRAYSCALE);

	int n = atoi(argv[2]);
	int q = atoi(argv[3]);

	if (n > min(img.rows, img.cols)){
		cout << "n exceeds the width or height of the image!" << endl;
		return -2;
	}

	simulateLowRes(img, n);

	if (q < 1 || q > 8){
		cout << "q needs to be in range [1, 8]!" << endl;
		return -3;
	}

	quantizeImg(img, q);

	//for (int i : {2, 4, 8}){
	//	simulateLowRes(img, i);
	//	quantizeImg(img, i);
	//}

	return 0;
}