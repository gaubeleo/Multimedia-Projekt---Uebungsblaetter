#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <time.h>
#include <direct.h>

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\ml\ml.hpp>

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

Mat drawSets(const Mat &img, const Mat &data, const Mat &labels)
{
	assert(img.type() == CV_8UC3);
	assert(data.type() == CV_32FC1);
	assert(labels.type() == CV_32FC1);
	assert(labels.rows == data.rows);
	assert(data.cols == 2 && labels.cols == 1);

	Mat newImg;
	img.copyTo(newImg);

	for (int i = 0; i < data.rows; i++)
	{
		float label = labels.at<float>(i);
		int y = (int)data.at<float>(i, 0);
		int x = (int)data.at<float>(i, 1);

		// -1. ^= pos label
		if (label == 1.)
			circle(newImg, Point(x, y), 3, Scalar(0, 255, 0), -1);
		else if (label == -1.)
			circle(newImg, Point(x, y), 3, Scalar(0, 0, 255), -1);
	}

	return newImg;
}

void createSets(Mat &data, Mat &labels, int margin, bool linearSperable)
{
	assert(data.type() == CV_32FC1);
	assert(labels.type() == CV_32FC1);
	assert(labels.rows == data.rows);
	assert(data.cols == 2 && labels.cols == 1);

	srand(time(NULL));
	for (unsigned i = 0; i < data.rows; i++)
	{
		int x, y;
		float label;
		if (linearSperable){
			x = rand() % 512 - (2 * margin);
			y = rand() % 512 - (2 * margin);
			if ((x + y) > 512-margin)
			{
				x += 2 * margin;
				y += 2 * margin;
			}
			label = (x + y) < 512 ? -1. : 1.;
		}
		else
		{
			x = rand() % 512;
			y = rand() % 512;
			label = (x + y) < 512 ? -1. : 1.;
			if ((x + y) > 512 - margin && (x + y) < 512 + margin)
				label = ((rand() % 2) * 2) - 1;
		}
		labels.at<float>(i) = label;

		data.at<float>(i, 0) = float(y);
		data.at<float>(i, 1) = float(x);
	}
}


void trainSVM(const char* filename, const Mat &data, const Mat &labels, int maxIter, bool linear)
{
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	if (linear)
		params.kernel_type = CvSVM::LINEAR;
	else
		params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, maxIter, 1e-8);

	CvSVM SVM;
	SVM.train_auto(data, labels, Mat(), Mat(), params);

	SVM.save(filename);
}

Mat visualizeSVM(char* filename, const Mat& canvas, const Mat& data, const Mat& labels)
{
	assert(canvas.type() == CV_8UC3);
	CvSVM SVM;
	SVM.load(filename);

	Mat newCanvas(canvas.rows, canvas.cols, canvas.type());
	for (int y = 0; y < newCanvas.rows; y++)
	{
		Vec3b *row = newCanvas.ptr<Vec3b>(y);
		for (int x = 0; x < newCanvas.cols; x++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << y, x);
			float response = SVM.predict(sampleMat, true);

			if (response < 0.)
			{
				if (response < -1.)
					row[x] = Vec3b(0, 80, 0);
				else
					row[x] = Vec3b(0, 120, 0);
			}
			else if (response >= 0)
			{
				if (response > 1.)
					row[x] = Vec3b(0, 0, 80);
				else
					row[x] = Vec3b(0, 0, 120);
			}
		}
	}

	return newCanvas;
}

int main(){
	int height = 512, width = 512, count = 100;
	Mat black_canvas(height, width, CV_8UC3, Scalar(0, 0, 0));

	// Aufgabe 4.1a)
	Mat data(count, 2, CV_32FC1), labels(count, 1, CV_32FC1);
	createSets(data, labels, 20, true);
	
	Mat &setCanvas = drawSets(black_canvas, data, labels);
	
	imshow("Pos and Neg Sets", setCanvas);
	waitKey();
	destroyAllWindows();

	// Aufgabe 4.1b)
	char* filename = "results\\linearHardSVM.xml";
	struct stat sb;
	if (!(stat("results", &sb) == 0 && sb.st_mode == S_IFDIR)){
		_mkdir("results");
	}

	trainSVM(filename, data, labels, 5000, true);
	// Aufgabe 4.1c)
	Mat SVMcanvas = visualizeSVM(filename, black_canvas, data, labels);
	SVMcanvas = drawSets(SVMcanvas, data, labels);

	imshow("SVM Canvas", SVMcanvas);
	waitKey();
	destroyAllWindows();

	// Aufagbe 4.2a)
	Mat data2(count, 2, CV_32FC1), labels2(count, 1, CV_32FC1);
	createSets(data2, labels2, 50, false);
	Mat &setCanvas2 = drawSets(black_canvas, data2, labels2);

	imshow("Soft Margin Set", setCanvas2);
	waitKey();
	destroyAllWindows();

	// Aufagbe 4.2b)
	filename = "results\\linearSoftSVM.xml";

	trainSVM(filename, data, labels, 30000, true);
	SVMcanvas = visualizeSVM(filename, black_canvas, data2, labels2);
	SVMcanvas = drawSets(SVMcanvas, data2, labels2);

	imshow("SVM Canvas 2", SVMcanvas);
	waitKey();
	destroyAllWindows();

	filename = "results\\rbfSoftSVM.xml";

	trainSVM(filename, data, labels, 30000, false);
	SVMcanvas = visualizeSVM(filename, black_canvas, data2, labels2);
	SVMcanvas = drawSets(SVMcanvas, data2, labels2);

	imshow("SVM Canvas RBF", SVMcanvas);
	waitKey();
	destroyAllWindows();


	return 0;
}
