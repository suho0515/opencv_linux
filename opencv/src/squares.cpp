// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv/MsgCenter.h"
#include "opencv/MsgAngle.h"
#include "opencv/MsgDetection.h"

#include "std_msgs/Float32.h"
#include <deque>

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

static const std::string OPENCV_WINDOW = "Squares detection";

static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage (it's got it all folks) to find\n"
    "squares in a list of images pic1-6.png\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares [file_name (optional)]\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;
}

int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

class Squares
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  vector<vector<Point> > squares;

  ros::Publisher ros_center;
  ros::Publisher ros_angle;
  ros::Publisher ros_detection;
  opencv::MsgCenter msgCenter;
  opencv::MsgAngle msgAngle;
  opencv::MsgDetection msgDetection;
  
  deque<bool> qDetection;
  int sumDetection;
  int avgDetection;

  deque<int> qAngle;
  int sumAngle;
  int avgAngle;

  deque<int> qCenterX;
  int sumCenterX;
  int avgCenterX;

  deque<int> qCenterY;
  int sumCenterY;
  int avgCenterY;

  int avg_filtering_fector;

public:
  Squares()
    : it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("/pylon_camera_node/image_raw", 1,
      &Squares::findSquares, this);
    image_pub_ = it_.advertise("/squares/output_video", 1);

    ros_center = nh_.advertise<opencv::MsgCenter>("center",100);
    ros_angle = nh_.advertise<opencv::MsgAngle>("angle",100);
    ros_detection = nh_.advertise<opencv::MsgDetection>("detection",100);
    
    

    cv::namedWindow(OPENCV_WINDOW);
  }

  ~Squares()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void findSquares(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
	
    // Squres Finding
    squares.clear();
    Mat pyr, timg, gray0(cv_ptr->image.size(), CV_8U), gray;
    // down-scale and upscale the image to filter out the noise
    pyrDown(cv_ptr->image, pyr, Size(cv_ptr->image.cols/2, cv_ptr->image.rows/2));
    pyrUp(pyr, timg, cv_ptr->image.size());
    vector<vector<Point> > contours;
    Point2f center;
    float radius;
    msgDetection.detection = 0;
    sumDetection = 0;
    
    


    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            /// Find the rotated rectangles and ellipses for each contour
            vector<RotatedRect> minRect( contours.size() );

             //for( int i = 0; i < contours.size(); i++ )
             //{
               //if( contours[i].size() > 5 )
                 //{ minEllipse[i] = fitEllipse( Mat(contours[i]) ); }
             //}

            vector<Point> approx;
            
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                minRect[i] = minAreaRect( approx ); // Function to get Angle

                minEnclosingCircle(approx,center, radius);
                
                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 5000 &&
                    fabs(contourArea(Mat(approx))) < 6000 &&
                    isContourConvex(Mat(approx)) &&
                    radius > 45 && radius < 60 &&
		    center.x >150 && center.x <400)
                {
                    
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 ) {		
                        squares.push_back(approx);

                        // Draw an example circle on the video stream
                        cv::circle(cv_ptr->image, cv::Point(center.x, center.y), 10, CV_RGB(255,0,0));
                        
                        //ROS_INFO("fabs(contourArea(Mat(approx))) : %f", fabs(contourArea(Mat(approx)))); 
                        //ROS_INFO("radius : %f", radius);
                        //ROS_INFO("every center : %f, %f", center.x, center.y);


			// Average Filtering for 'detection', 'center', 'angle'.

                        msgAngle.angle=-(minRect[i].angle);

			msgCenter.x=center.x;
                        msgCenter.y=center.y;
			
			/*
			avg_filtering_fector = 20;			

			if(qAngle.size() < avg_filtering_fector) {
				qAngle.push_back(msgAngle.angle);
				qCenterX.push_back(msgCenter.x);
				qCenterY.push_back(msgCenter.y);
			} 
			else if(qAngle.size() >= avg_filtering_fector) {
				qAngle.pop_front();
				qCenterX.pop_front();
				qCenterY.pop_front();

				qAngle.push_back(msgAngle.angle);	
				qCenterX.push_back(msgCenter.x);
				qCenterY.push_back(msgCenter.y);

				sumAngle = 0;
				sumCenterX = 0;
				sumCenterY = 0;

				avgAngle = 0;
				avgCenterX = 0;
				avgCenterY = 0;

				for(int i=0;i<avg_filtering_fector;i++)
				{
					sumAngle += qAngle[i];
					sumCenterX += qCenterX[i];
					sumCenterY += qCenterY[i];
				}

				avgAngle = sumAngle/avg_filtering_fector;
				avgCenterX = sumCenterX/avg_filtering_fector;
				avgCenterY = sumCenterY/avg_filtering_fector;

				msgAngle.angle = avgAngle;
				msgCenter.x = avgCenterX;
				msgCenter.y = avgCenterY;
				*/

				msgDetection.detection++;
				
				ros_angle.publish(msgAngle);
				ros_center.publish(msgCenter);

				
				//ROS_INFO("angle : %f", msgAngle.angle);
				//ROS_INFO("center : %f, %f", msgCenter.x, msgCenter.y);
			//}
			
                        //ROS_INFO("cv_ptr->image.cols : %d", cv_ptr->image.cols);
                        //ROS_INFO("cv_ptr->image.rows : %d", cv_ptr->image.rows);

                    }
                }

            }
        }
    }

    // the function draws all the squares in the image
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(cv_ptr->image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    
    /*
    if(msgDetection.detection > 1) {
	if(qDetection.size() < 10) {
		qDetection.push_back(true);
	} 
	else if(qDetection.size() >= 10) {
		qDetection.pop_front();
		qDetection.push_back(true);
	}
    }
    else {
	if(qDetection.size() < 10) {
		qDetection.push_back(false);
	} 
	else if(qDetection.size() >= 10) {
		qDetection.pop_front();
		qDetection.push_back(false);
	}
    }

    for (int i=0; i<10; i++) {
    	//cout << "detection[" << i << "] : " << qDetection[i] << endl;
	sumDetection += qDetection[i];  
    }

    if(sumDetection > 8) {
	ros_detection.publish(msgDetection);
	//ROS_INFO("detection : %d", msgDetection.detection);

	ros_angle.publish(msgAngle);
	ros_center.publish(msgCenter);

	// Draw an example circle on the video stream
        cv::circle(cv_ptr->image, cv::Point(msgCenter.x, msgCenter.y), 10, CV_RGB(255,0,0));
    }
    */

    ros_detection.publish(msgDetection);

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

    // Output modified video stream
    image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "squares");
  Squares sq;
  ros::spin();
  return 0;
}

