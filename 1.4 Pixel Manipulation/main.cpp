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
	Mat lowResImg(img.rows, img.cols, CV_8UC1);

	for (int y = 0; y < img.rows; y += n){
		for (int x = 0; x < img.cols; x += n){
			int value = 0;
			for (int yi = 0; yi < n; yi++){
				for (int xi = 0; xi < n; xi++){
					value += img.at<uchar>(y, x);
				}
			}
			for (int yi = 0; yi < n; yi++){
				for (int xi = 0; xi < n; xi++){
					lowResImg.at<uchar>(y, x) = value / (n*n);
				}
			}
		}
	}

	return lowResImg;
}

Mat quantizeImg(Mat img, int q){
	Mat quantizedImg(img.rows, img.cols, CV_8UC1);

	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			uchar value = img.at<uchar>(y, x);
			uchar newValue = ((value >> (8 - q)) << (8 - q)) + 256 / (1 << q + 1);
			quantizedImg.at<uchar>(y, x) = newValue;
		}
	}

	return quantizedImg;
}

int main(int argc, char* argv[]) {
	/*if (argc != 4){
		cout << "this program needs to be run with three command line parameters: <filename> <int> <int>" << endl;
		cout << "example: '1.4 Pixel Manipulation.exe' 'src\\lenna.jpg' 3 3" << endl;
		return -1;
	}*/

	string dir = "", filename = "";

	string fullFilename(argv[1]);

	dir = fullFilename.substr(0, fullFilename.rfind("\\"));
	//filename = fullFilename.substr(fullFilename.rfind("\\")+1, fullFilename.length);

	cout << dir << filename << endl;

	return 0;

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

	return 0;
}