# Depth Sensor Using Orbbec SDK for Facial Detection - using JSON file import

# Prerequisite
1. Visual Studio 2013 or higher
2. [Orbbec SDK](https://orbbec3d.com/develop/)
3. [Sensor Driver](https://www.dropbox.com/s/oyw9pcfrf0zgck1/Driver_Windows.zip?dl=0)
4. External Libraries
  1. [OpenCV](https://opencv.org/releases.html)
  2. [Nlohmann JSON](https://github.com/nlohmann/json)

Before we can implement facial detection in their code, setting up the environment
would be required.

Simply download Visual Studio and install C++ workload. If error occurs when building
the Orbbec SDK, the following link might help.

https://stackoverflow.com/questions/42777424/visual-studio-2017-errors-on-standard-headers

https://developercommunity.visualstudio.com/content/problem/48806/cant-find-v140-in-visual-studio-2017.html

Orbbec SDK is straightforward - download and extract. Open into "samples" -> "vs2015"
and open the .sln file.

Setting up the sensor driver is also straightforward, just follow their instruction.

To setup the libraries for visual studio environment, go to "Project" -> "Properties",
in "C/C++" -> "Additional Include Directories" add nlohmann "single_include" and
opencv "build/include". In "Linker" -> "Additional Library Directories" add opencv
"build/x64/vc14/lib" and also in "Linker" -> "Input" -> "Additional Dependencies"
add "opencv_worldXXX.lib". Finally add opencv_worldXXX.dll at the execution file
(samples\vs2015\bin\Release)

Now it should be able to build and run.

To change to other program to run, right-click on project in the "Solution Explorer"
and click "Set as Startup Project". This project will be working on the "SimpleDepthViewer-SFML"
which will give us the depth of the image. Since there is no colour, we will be
importing "SimpleColorViewer-SFML" to grab the colour of the pixel. The main usage
of the SDK is to grab the depth of each pixel and colour (which could be extracted with opencv as well).

Most processes will be completed through OpenCV and Nlohmann JSON to detect face
at a certain distance.

In "SimpleDepthViewer", it will contain the following:

    // depth viewer
    #include <SFML/Graphics.hpp>
    #include <astra/astra.hpp>
    #include "LitDepthVisualizer.hpp"
    #include <chrono>
    #include <iostream>
    #include <iomanip>
    #include <key_handler.h>
    #include <sstream>

Add the additional libraries to handle face detection and json.

    // colour viewer
    #include <SFML/Graphics.hpp>
    #include <astra_core/astra_core.hpp>

    //extra
    #include <thread>
    #include <conio.h>
    #include <string>
    #include <cmath>
    #include <fstream>


    // opencv
    #include <opencv2/objdetect.hpp>
    #include <opencv2/opencv.hpp>

    // json
    #include <nlohmann/json.hpp>


    using namespace std;
    using json = nlohmann::json;

To open a json file:

    // loading json file
    std::ifstream ifs("setting.json");
    json j = json::parse(ifs);

And to load the json file to a variable setting:

    // setting
    int minDist = j["minDist"];
    int maxDist = j["maxDist"];
    int timerTrigger = j["timer"];
    int Xdepth = j["Xdepth"]; // x-axis
    int Ydepth = j["Ydepth"]; // y-axis
    int windowXSize = Xdepth * 2; // x-dimension
    int windowYSize = Ydepth * 2; // y-dimension

Global Variable are used to instead of passing local scope into functions.

    // global variables
    cv::Mat DisplayImage = cv::Mat::zeros(cv::Size(Xdepth, Ydepth), CV_8UC3);
    cv::Mat OriginalImage = cv::Mat::zeros(cv::Size(Xdepth, Ydepth), CV_8UC3);
    int distanceValue[640][480] = { 0 };
    vector<cv::Rect> objectsDetected;
    vector<cv::Mat> faceSaved;
    vector<bool> triggerDetection;
    vector<bool> triggeredFace;
    vector<int> savedTime;
    int displayTimer = 0;
    int countFace = 0;
    int prevCount = 0;
    int countFaceTriggered = 0;
    bool colourData = false;

Afterward, "SimpleColorViewer-SFML" will be imported across to get the colour of
the image and this following code is added in "on_frame_ready" function, below "const
astra::RgbPixel* colorData = colorFrame.data()":

    if (colourData) {
      for (int i = 0; i < width*height; i++)
      {
        int index = i % width + width * (i / width);
        DisplayImage.at<uchar>(i / width, 3 * (i%width)) = colorData[index].b;
        DisplayImage.at<uchar>(i / width, 3 * (i%width) + 1) = colorData[index].g;
        DisplayImage.at<uchar>(i / width, 3 * (i%width) + 2) = colorData[index].r;
      }
      OriginalImage.clone();
    }
    colourData = true;

'colourData' is initially set to false to to skip the first assignment as the screen
would be blank - causing an error. After the first cycle, it will grab the colour
details of each pixels. This is the only change the import will have, the rest
remains the same.

With the "SimpleDepthViewer-SFML", there's a few section removed as it is not required.
The most important section is the "update_mouse_position" function. This function
is replaced with the following:

    void update_depth(sf::RenderWindow& window, const astra::CoordinateMapper& coordinateMapper) {
      // aim: gathering distance value
      for (int x640 = 0; x640 < 160; x640++) { // based on depth 160 or 640
        for (int y480 = 0; y480 < 120; y480++) {
          mouseX_ = x640;
          mouseY_ = y480;
          if (mouseX_ >= depthWidth_ ||
              mouseY_ >= depthHeight_ ||
              mouseX_ < 0 ||
              mouseY_ < 0) {
              return;
          }
          const size_t index = (depthWidth_ * mouseY_ + mouseX_);
          const short z = depthData_[index];
          coordinateMapper.convert_depth_to_world(float(mouseX_), float(mouseY_), float(z), mouseWorldX_, mouseWorldY_, mouseWorldZ_);
          distanceValue[x640][y480] = mouseWorldZ_; // recording distance value

        }
      }
    }

To handle face detection, another function is created:

    void detectAndDraw(cv::Mat& frame) {

where it will load the facial detection data:

    double scale = 1
    cv::CascadeClassifier face_cascade;
    face_cascade.load("C:\\C++ External Libraries\\opencv_3_4_5\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");

afterward, setup time trigger:

    time_t timer;
    time(&timer);  /* get current time; same as: timer = time(NULL)  */
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

To detect face:

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

As soon as there is a change of number of face, it will add to count:

    int value = prevCount - faces.size();
      if (value < 0) {
      countFace = countFace + (int)abs((double)prevCount - (double)faces.size());
    }
    prevCount = faces.size();

There will be a initial setup to grab all the faces detected:

    if (objectsDetected.size() == 0) { // first run through
      for (size_t i = 0; i < faces.size(); i++)
      {
        cv::Rect r = faces[i];
        if (distanceValue[(r.x + r.width / 2) / 4][(r.y + r.height / 2) / 4] > minDist && distanceValue[(r.x + r.width / 2) / 4][(r.y + r.height / 2) / 4] < maxDist) { // divide by 4 since depth viewer is 4x smaller in dimension
          objectsDetected.push_back(r); // the ROIs
          faceSaved.push_back(OriginalImage(r)); // the cropped image
          savedTime.push_back(difftime(timer, mktime(&y2k))); // the time spotted
          triggerDetection.push_back(false); // triggering detection
          triggeredFace.push_back(false); // triggering detection
        }
      }
    }

Using vectors setup as global variable, each vector of a specific index contain
the detail of that image.

When there are faces stored, it will run the following:

      else { // consecutive run through
        if (difftime(timer, mktime(&y2k)) - displayTimer > 1) {
          cout << "number of face currently detected: " << faces.size()
            //<< "\tnumber of unique face detected: " << objectsDetected.size()
            //<< "\tface reappearing: " << countFace
            << "\tface triggered: " << countFaceTriggered
            << endl;
          displayTimer = difftime(timer, mktime(&y2k));
        }

        for (int x = 0; x < objectsDetected.size(); x++) {
          triggerDetection.at(x) = false;
        }
        for (size_t i = 0; i < faces.size(); i++) {
          cv::Rect r = faces[i];
          cv::Scalar color = cv::Scalar(0, 255, 0);
          if (distanceValue[(r.x + r.width / 2) / 4][(r.y + r.height / 2) / 4] > minDist && distanceValue[(r.x + r.width / 2) / 4][(r.y + r.height / 2) / 4] < maxDist) {
            rectangle(frame, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)), cvPoint(cvRound((r.x +
              r.width - 1)*scale), cvRound((r.y + r.height - 1)*scale)), color, 3, 8, 0);
            bool matched = false;
            for (int x = 0; x < objectsDetected.size(); x++)
            { // find matching
              if (abs(objectsDetected.at(x).height - r.height) < 75.00 && abs(objectsDetected.at(x).width - r.width) < 75.00) { // find similarity - not really possible
                matched = true;
                objectsDetected.at(x) = r; // re-save the image details
                faceSaved.at(x) = OriginalImage(r);
                triggerDetection.at(x) = true;
                if (difftime(timer, mktime(&y2k)) - savedTime.at(x) > timerTrigger) { // timer trigger
                                                    //cout << "highlighting detection of " << x << " index" << endl;
                  cv::Scalar color = cv::Scalar(255, 0, 0);
                  rectangle(frame, cvPoint(cvRound(objectsDetected.at(x).x*scale), cvRound(objectsDetected.at(x).y*scale)), cvPoint(cvRound((objectsDetected.at(x).x +
                    objectsDetected.at(x).width - 1)*scale), cvRound((objectsDetected.at(x).y + objectsDetected.at(x).height - 1)*scale)), color, 3, 8, 0);
                  if (!triggeredFace.at(x)) {
                    countFaceTriggered++;
                    triggeredFace.at(x) = true;
                    cout << "object detected at " << distanceValue[(r.x + r.width / 2) / 4][(r.y + r.height / 2) / 4] << " mm away from the camera" << endl;
                  }
                }
                break;
              }
            }
            if (!matched) { // new unique face
              objectsDetected.push_back(r); // the ROIs
              faceSaved.push_back(OriginalImage(r)); // the cropped image
              savedTime.push_back(difftime(timer, mktime(&y2k))); // the time spotted
              triggerDetection.push_back(false); // triggering detection
              triggeredFace.push_back(false); // triggering detection
              //cout << "new face detected" << endl;

            }
          }
        }

        for (int x = 0; x < triggerDetection.size(); x++) {
          //cout << x << " index is " << triggerDetection.at(x) << endl;
          if (!triggerDetection.at(x)) {
            //	cout << "reset time for index " << x << endl;
            savedTime.at(x) = difftime(timer, mktime(&y2k));
            triggeredFace.at(x) = false;
          }
        }
      }
      imshow("Detected Face", frame);
    } // end of the function

In the main function, the "SimpleColorViewer-SFML" has to be setup:

    sf::RenderWindow windowColour(sf::VideoMode(windowXSize, windowYSize), "Color Viewer");

    astra::StreamSet streamSetColour;
    astra::StreamReader readerColour = streamSetColour.create_reader();

    readerColour.stream<astra::ColorStream>().start();

    ColorFrameListener listenerColour;
    readerColour.add_listener(listenerColour);

During the loop to capture each frame, it is changed to the following:

    while (windowColour.isOpen())
    {
      astra_update();

      sf::Event event;
      while (windowColour.pollEvent(event))
      {
      	switch (event.type)
      	{
      	case sf::Event::Closed:
      		windowColour.close();
      		windowDepth.close();

      		break;
      	case sf::Event::KeyPressed:
      	{
      		if (event.key.code == sf::Keyboard::Escape ||
      			(event.key.code == sf::Keyboard::C && event.key.control))
      		{
      			windowColour.close();
      			windowDepth.close();

      		}
      	}
      	default:
      		break;
      	}
      }

      // clear the window with black color
      windowColour.clear(sf::Color::Black);
      windowDepth.clear(sf::Color::Black);

      listenerColour.drawTo(windowColour);
      listenerDepth.draw_to(windowDepth);

      windowColour.display();
      windowDepth.display();

      auto coordinateMapper = depthStream.coordinateMapper();
      listenerDepth.update_depth(windowDepth, coordinateMapper);


      if (!shouldContinue)
      {
      	windowColour.close();

      }
      detectAndDraw(DisplayImage);
    }

The created function is placed at the end to begin detecting the faces.

The full code will be uploaded separately.
