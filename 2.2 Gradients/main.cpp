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

Mat loadImg(string directory, string filename){
	string fullFilename = string(directory + "\\" + filename);
	Mat image;

	image = imread(fullFilename, CV_LOAD_IMAGE_COLOR);

	if (!image.data){
		cout << "image file '" << fullFilename << "' could not be opened" << endl;
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

void displayChannels(Mat img){
	string name;
	vector<Mat> channels;

	//split(img, channels);

	for (int i = 0; i < 3; i++){
		channels.push_back(Mat(img.rows, img.cols, CV_8UC1));
	}

	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			Vec3b pixel = img.at<Vec3b>(y, x);
			channels[0].at<uchar>(Point(x, y)) = pixel[0];
			channels[1].at<uchar>(Point(x, y)) = pixel[1];
			channels[2].at<uchar>(Point(x, y)) = pixel[2];
		}
	}

	// Reverse because OPEN_CV uses BGR instead of RGB
	imshow("RED", channels[2]);
	imshow("GREEN", channels[1]);
	imshow("BLUE", channels[0]);

	waitKey();
	destroyAllWindows();
}

void reverseChannels(Mat img){
	Mat reversedImg = Mat(img.rows, img.cols, CV_8UC3);

	for (int y = 0; y < img.rows; y++){
		for (int x = 0; x < img.cols; x++){
			Vec3b pixel = img.at<Vec3b>(y, x);
			reversedImg.at<Vec3b>(Point(x, y)) = Vec3b(pixel[2], pixel[1], pixel[0]);
		}
	}

	saveImg("results", "BGR_img.jpg", reversedImg);
}

int main(){
	Mat img = loadImg("src", "DSC_0078.jpg");

	// Aufgabe a)
	displayChannels(img);

	// Aufgabe b)

	//Aufgabe c)
	reverseChannels(img);

	//Aufgabe d)


	return 0;
}