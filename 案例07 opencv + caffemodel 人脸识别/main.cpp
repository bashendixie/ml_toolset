#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace dnn;

int main(int argc, char** argv)
{
    Net net = readNetFromCaffe("C:/Users/zyh/Desktop/deploy.prototxt",
        "C:/Users/zyh/Desktop/res10_300x300_ssd_iter_140000.caffemodel");
    Mat image = imread("C:/Users/zyh/Desktop/5.jpg");
    Mat image1;
    //resize(image, image1, Size(300, 300));
    //Mat blob = blobFromImage(image1, 1, Size(300, 300), Scalar(104, 117, 123));
    Mat blob = blobFromImage(image, 1, Size(), Scalar(104, 117, 123));

    net.setInput(blob);
    Mat detections = net.forward();
    Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        if (detectionMat.at<float>(i, 2) >= 0.13)
        {
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

            Rect object((int)xLeftBottom, (int)yLeftBottom,
                (int)(xRightTop - xLeftBottom),
                (int)(yRightTop - yLeftBottom));

            rectangle(image, object, Scalar(0, 255, 0));
        }
    }


    //œ‘ æÕº∆¨
    imshow("img", image);
    waitKey(0);
    return 0;
}
