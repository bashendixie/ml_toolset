// 学习opencv加载Caffe模型

#include <opencv.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/trace.hpp>

#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

static void getMaxClass(const Mat& probBlob, int* classId, double* classProb)
{
	Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;
	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

static std::vector<String> readClassNames(const char* filename = "C:/Users/zyh/Desktop/dnn/synset_words.txt")
{
	std::vector<String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}

int main2(int argc, char** argv)
{
	Net net = readNetFromCaffe("C:/Users/zyh/Desktop/dnn/bvlc_googlenet.prototxt", "C:/Users/zyh/Desktop/dnn/bvlc_googlenet.caffemodel");
	Mat image = imread("C:/Users/zyh/Desktop/feiji.jpg");
	Mat inputBlob = blobFromImage(image, 1, Size(224, 224), Scalar(104, 117, 123));
	Mat prob;
	cv::TickMeter t;
	for (int i = 0; i < 10; i++)
	{
		CV_TRACE_REGION("forward");
		net.setInput(inputBlob, "data");        //set the network input
		t.start();
		prob = net.forward("prob");                          //compute output
		t.stop();
	}
	int classId;
	double classProb;
	getMaxClass(prob, &classId, &classProb);//find the best class
	std::vector<String> classNames = readClassNames();

	string text = classNames.at(classId) + to_string(classProb * 100);

	putText(image, text, Point(5, 25), FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);

	std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
	std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
	std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;

	imshow("Image", image);
	waitKey(0);
	//system("pause");
	return 0;
}
