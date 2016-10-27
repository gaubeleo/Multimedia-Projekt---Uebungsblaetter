#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <time.h>

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
		cout << "image file " << fullFilename << " could not be opened" << endl;
		getchar();
		exit(-1);
	}

	return image;
}

int main(){
	//loadImag();

	return 0;
}