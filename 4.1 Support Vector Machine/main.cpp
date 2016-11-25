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



/////////////////////////////////////////////////////////////////////////////

Mat drawSets(const Mat &img, const vector<Point_<int>> posSet, const vector<Point_<int>>& negSet)
{
	cout << posSet.size();
	assert(img.type() == CV_8UC3);
	Mat newImg;
	img.copyTo(newImg);

	for (Point p : posSet)
	{
		circle(newImg, p, 3, Scalar(0, 255, 0), -1);
	}
	for (Point p : negSet)
	{
		circle(newImg, p, 3, Scalar(0, 0, 255), -1);
	}

	return newImg;
}

void createLinearSeperableSet(vector<Point_<int>> &posSet, vector<Point_<int>> &negSet, unsigned posCount, unsigned negCount)
{
	srand(time(NULL));
	for (unsigned i = 0; i < posCount; i++)
	{
		Point p(rand() % 250, rand() % 512);
		posSet.push_back(p);
	}
	for (unsigned i = 0; i < negCount; i++)
	{
		Point p(512-(rand() % 250), rand() % 512);
		negSet.push_back(p);
	}
}

int main(){
	int height = 512, width = 512;
	Mat canvas(height, width, CV_8UC3, Scalar(0, 0, 0));

	// Aufgabe 4.1a)
	vector<Point> posSet, negSet;
	createLinearSeperableSet(posSet, negSet, 30, 25);
	Mat &setCanvas = drawSets(canvas, posSet, negSet);
	imshow("Pos and Neg Sets", setCanvas);
	waitKey();
	destroyAllWindows();

	// Aufgabe 4.1b)


	return 0;
}
