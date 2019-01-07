// This file is part of the Orbbec Astra SDK [https://orbbec3d.com]
// Copyright (c) 2015-2017 Orbbec 3D
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Be excellent to each other.


// depth viewer
#include <SFML/Graphics.hpp>
#include <astra/astra.hpp>
#include "LitDepthVisualizer.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <key_handler.h>
#include <sstream>

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

// loading json file
std::ifstream ifs("setting.json");
json j = json::parse(ifs);

// setting
int minDist = j["minDist"];
int maxDist = j["maxDist"];
int timerTrigger = j["timer"];
int Xdepth = j["Xdepth"]; // x-axis
int Ydepth = j["Ydepth"]; // y-axis
int windowXSize = Xdepth * 2; // x-dimension
int windowYSize = Ydepth * 2; // y-dimension


// global variables
cv::Mat DisplayImage = cv::Mat::zeros(cv::Size(Xdepth, Ydepth), CV_8UC3); // retrieving the pixel RGB 
int distanceValue[640][480] = { 0 };
bool colourData = false;
int displayTimer = 0;


vector<cv::Rect> faces_verifying;
vector<bool> faces_verifyingExist;
vector<int> faces_verifyingStartTime;
vector<int> faces_verifyingLastSeen;

vector<cv::Rect> faces_tracking;
vector<bool> faces_trackingExist;
vector<int> faces_trackingStartTime;
vector<int> faces_trackingLastSeen;

vector<int> faces_IndexDel;

int numberOfFaces = 0;


class ColorFrameListener : public astra::FrameListener
{
public:
	ColorFrameListener()
	{
		prev_ = ClockType::now();
	}

	void init_texture(int width, int height)
	{
		if (displayBuffer_ == nullptr || width != displayWidth_ || height != displayHeight_)
		{
			displayWidth_ = width;
			displayHeight_ = height;

			// texture is RGBA
			int byteLength = displayWidth_ * displayHeight_ * 4;

			displayBuffer_ = BufferPtr(new uint8_t[byteLength]);
			std::memset(displayBuffer_.get(), 0, byteLength);

			texture_.create(displayWidth_, displayHeight_);
			sprite_.setTexture(texture_, true);
			sprite_.setPosition(0, 0);
		}
	}

	virtual void on_frame_ready(astra::StreamReader& reader, astra::Frame& frame) override
	{
		const astra::ColorFrame colorFrame = frame.get<astra::ColorFrame>();

		int width = colorFrame.width();
		int height = colorFrame.height();

		init_texture(width, height);

		const astra::RgbPixel* colorData = colorFrame.data();

		// getting colour image
		if (colourData) {
			for (int i = 0; i < width*height; i++)
			{
				int index = i % width + width * (i / width);
				DisplayImage.at<uchar>(i / width, 3 * (i%width)) = colorData[index].b;
				DisplayImage.at<uchar>(i / width, 3 * (i%width) + 1) = colorData[index].g;
				DisplayImage.at<uchar>(i / width, 3 * (i%width) + 2) = colorData[index].r;
			}
		}
		colourData = true;


		for (int i = 0; i < width * height; i++)
		{

			int rgbaOffset = i * 4;
			displayBuffer_[rgbaOffset] = colorData[i].r;
			displayBuffer_[rgbaOffset + 1] = colorData[i].g;
			displayBuffer_[rgbaOffset + 2] = colorData[i].b;
			displayBuffer_[rgbaOffset + 3] = 255;
		}
		texture_.update(displayBuffer_.get());
	}

	void drawTo(sf::RenderWindow& window)
	{
		if (displayBuffer_ != nullptr)
		{
			float imageScale = window.getView().getSize().x / displayWidth_;
			sprite_.setScale(imageScale, imageScale);
			window.draw(sprite_);
		}
	}

private:
	using DurationType = std::chrono::milliseconds;
	using ClockType = std::chrono::high_resolution_clock;

	ClockType::time_point prev_;
	float elapsedMillis_{ .0f };

	sf::Texture texture_;
	sf::Sprite sprite_;

	using BufferPtr = std::unique_ptr<uint8_t[]>;
	BufferPtr displayBuffer_{ nullptr };

	int displayWidth_{ 0 };
	int displayHeight_{ 0 };

	using buffer_ptr = std::unique_ptr<astra::RgbPixel[]>;
	buffer_ptr buffer_;
	unsigned int lastWidth_;
	unsigned int lastHeight_;

};


class DepthFrameListener : public astra::FrameListener
{
public:
	DepthFrameListener()
	{
		prev_ = ClockType::now();
		font_.loadFromFile("Inconsolata.otf");
	}

	void init_texture(int width, int height)
	{
		if (!displayBuffer_ ||
			width != displayWidth_ ||
			height != displayHeight_)
		{
			displayWidth_ = width;
			displayHeight_ = height;

			// texture is RGBA
			const int byteLength = displayWidth_ * displayHeight_ * 4;

			displayBuffer_ = BufferPtr(new uint8_t[byteLength]);
			std::fill(&displayBuffer_[0], &displayBuffer_[0] + byteLength, 0);

			texture_.create(displayWidth_, displayHeight_);
			sprite_.setTexture(texture_, true);
			sprite_.setPosition(0, 0);
		}
	}


	void on_frame_ready(astra::StreamReader& reader,
		astra::Frame& frame) override
	{
		const astra::PointFrame pointFrame = frame.get<astra::PointFrame>();
		const int width = pointFrame.width();
		const int height = pointFrame.height();

		init_texture(width, height);

		copy_depth_data(frame);

		visualizer_.update(pointFrame);

		const astra::RgbPixel* vizBuffer = visualizer_.get_output();

		for (int i = 0; i < width * height; i++)
		{
			const int rgbaOffset = i * 4;
			displayBuffer_[rgbaOffset] = vizBuffer[i].r;
			displayBuffer_[rgbaOffset + 1] = vizBuffer[i].b;
			displayBuffer_[rgbaOffset + 2] = vizBuffer[i].g;
			displayBuffer_[rgbaOffset + 3] = 255;
		}

		texture_.update(displayBuffer_.get());
	}

	void copy_depth_data(astra::Frame& frame)
	{
		const astra::DepthFrame depthFrame = frame.get<astra::DepthFrame>();

		if (depthFrame.is_valid())
		{
			const int width = depthFrame.width();
			const int height = depthFrame.height();
			if (!depthData_ || width != depthWidth_ || height != depthHeight_)
			{
				depthWidth_ = width;
				depthHeight_ = height;

				// texture is RGBA
				const int byteLength = depthWidth_ * depthHeight_ * sizeof(uint16_t);

				depthData_ = DepthPtr(new int16_t[byteLength]);
			}

			depthFrame.copy_to(&depthData_[0]);
		}
	}

	// ------------------------- objects detection of the middle section ------------------------- //
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

	void draw_to(sf::RenderWindow& window)
	{
		if (displayBuffer_ != nullptr)
		{
			const float depthWScale = window.getView().getSize().x / displayWidth_;
			const float depthHScale = window.getView().getSize().y / displayHeight_;

			sprite_.setScale(depthWScale, depthHScale);
			window.draw(sprite_);

		}
	}


private:
	samples::common::LitDepthVisualizer visualizer_;

	using DurationType = std::chrono::milliseconds;
	using ClockType = std::chrono::high_resolution_clock;

	ClockType::time_point prev_;
	float elapsedMillis_{ .0f };

	sf::Texture texture_;
	sf::Sprite sprite_;
	sf::Font font_;

	int displayWidth_{ 0 };
	int displayHeight_{ 0 };

	using BufferPtr = std::unique_ptr<uint8_t[]>;
	BufferPtr displayBuffer_{ nullptr };

	int depthWidth_{ 0 };
	int depthHeight_{ 0 };

	using DepthPtr = std::unique_ptr<int16_t[]>;
	DepthPtr depthData_{ nullptr };

	int mouseX_{ 0 };
	int mouseY_{ 0 };
	float mouseWorldX_{ 0 };
	float mouseWorldY_{ 0 };
	float mouseWorldZ_{ 0 };

};

astra::DepthStream configure_depth(astra::StreamReader& reader)
{
	auto depthStream = reader.stream<astra::DepthStream>();

	auto oldMode = depthStream.mode();

	//We don't have to set the mode to start the stream, but if you want to here is how:
	astra::ImageStreamMode depthMode;

	depthMode.set_width(160); // changes things - be careful
	depthMode.set_height(120);
	depthMode.set_pixel_format(astra_pixel_formats::ASTRA_PIXEL_FORMAT_DEPTH_MM);
	depthMode.set_fps(30);

	depthStream.set_mode(depthMode);

	auto newMode = depthStream.mode();
	printf("Changed depth mode: %dx%d @ %d -> %dx%d @ %d\n",
		oldMode.width(), oldMode.height(), oldMode.fps(),
		newMode.width(), newMode.height(), newMode.fps());

	return depthStream;
}


// face detection
void detectAndDraw(cv::Mat& frame) {

	// loading in face detection
	double scale = 1;
	cv::CascadeClassifier face_cascade;
	face_cascade.load("C:\\C++ External Libraries\\opencv_3_4_5\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml");

	// setting up current time access
	time_t timer;
	time(&timer);  /* get current time; same as: timer = time(NULL)  */
	struct tm y2k = { 0 };
	y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
	y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;


	// load all detected faces into 'faces' vector
	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));


	// display message
	if (difftime(timer, mktime(&y2k)) - displayTimer > 1) {
		std::cout << "Current count: " << numberOfFaces << std::endl;
		displayTimer = difftime(timer, mktime(&y2k));
	}


	// all tracked and verifying have to be detected again
	// dealing with verifying faces
	for (int i = 0; i < faces_verifyingExist.size(); i++)
	{
		faces_verifyingExist[i] = false;
		// loop through all faces available
		for (int j = 0; j < faces.size(); j++)
		{
			// detected in boundary -> exist
			if (!(faces[j].x + faces[j].width < faces_verifying[i].x) &&
				!(faces[j].x > faces_verifying[i].x + faces_verifying[i].width) &&
				!(faces[j].y + faces[j].height < faces_verifying[i].y) &&
				!(faces[j].y > faces_verifying[i].y + faces_verifying[i].height)
				)
			{
				faces_verifyingExist[i] = true;
				cv::Scalar color = cv::Scalar(0, 255, 0);
				rectangle(frame, cvPoint(cvRound(faces_verifying.at(i).x*scale), cvRound(faces_verifying.at(i).y*scale)), cvPoint(cvRound((faces_verifying.at(i).x +
					faces_verifying.at(i).width - 1)*scale), cvRound((faces_verifying.at(i).y + faces_verifying.at(i).height - 1)*scale)), color, 3, 8, 0);
				// adjust new values
				faces_verifying[i] = faces[j];
				faces_verifyingLastSeen[i] = difftime(timer, mktime(&y2k));

				// if exist -> look at time -> put in tracking and remove from verifying
				if (difftime(timer, mktime(&y2k)) - faces_verifyingStartTime.at(i) > 2)
				{
					faces_IndexDel.push_back(i);

					faces_tracking.push_back(faces_verifying.at(i));
					faces_trackingLastSeen.push_back(difftime(timer, mktime(&y2k)));
					faces_trackingStartTime.push_back(difftime(timer, mktime(&y2k)));
					faces_trackingExist.push_back(true);
				}
			}
			break;
		}

		// does not exist -> look at time
		if (!faces_verifyingExist[i])
		{
			// last seen more than 0.5s ago and start verifying more than 2s ago -> delete
			if (difftime(timer, mktime(&y2k)) - faces_verifyingStartTime.at(i) > 2 &&
				difftime(timer, mktime(&y2k)) - faces_verifyingLastSeen.at(i) > 0.5)
			{
				faces_IndexDel.push_back(i);
			}
		}
	}
	// reverse delete to not mess up indexing before deleting
	std::reverse(faces_IndexDel.begin(), faces_IndexDel.end());
	for (int i = 0; i < faces_IndexDel.size(); i++)
	{
		faces_verifying.erase(faces_verifying.begin() + faces_IndexDel.at(i));
		faces_verifyingExist.erase(faces_verifyingExist.begin() + faces_IndexDel.at(i));
		faces_verifyingLastSeen.erase(faces_verifyingLastSeen.begin() + faces_IndexDel.at(i));
		faces_verifyingStartTime.erase(faces_verifyingStartTime.begin() + faces_IndexDel.at(i));
	}
	// clear deleting
	faces_IndexDel.clear();


	// similarly dealing with tracking faces
	for (int i = 0; i < faces_trackingExist.size(); i++)
	{
		faces_trackingExist[i] = false;
		// loop through all faces available
		for (int j = 0; j < faces.size(); j++)
		{
			// detected in boundary -> exist
			if (!(faces[j].x + faces[j].width < faces_tracking[i].x) &&
				!(faces[j].x > faces_tracking[i].x + faces_tracking[i].width) &&
				!(faces[j].y + faces[j].height < faces_tracking[i].y) &&
				!(faces[j].y > faces_tracking[i].y + faces_tracking[i].height)
				)
			{
				faces_trackingExist[i] = true;
				cv::Scalar color = cv::Scalar(255, 0, 0);
				rectangle(frame, cvPoint(cvRound(faces_tracking.at(i).x*scale), cvRound(faces_tracking.at(i).y*scale)), cvPoint(cvRound((faces_tracking.at(i).x +
					faces_tracking.at(i).width - 1)*scale), cvRound((faces_tracking.at(i).y + faces_tracking.at(i).height - 1)*scale)), color, 3, 8, 0);
				// adjust new values
				faces_tracking[i] = faces[j];
				faces_trackingLastSeen[i] = difftime(timer, mktime(&y2k));
			}
			break;
		}

		// does not exist -> look at time
		if (!faces_trackingExist[i])
		{
			// last seen more than 0.5s ago and start verifying more than 2s ago -> delete
			if (difftime(timer, mktime(&y2k)) - faces_trackingStartTime.at(i) > 2 &&
				difftime(timer, mktime(&y2k)) - faces_trackingLastSeen.at(i) > 2)
			{
				faces_IndexDel.push_back(i);
			}
		}
	}
	// reverse delete to not mess up indexing before deleting
	std::reverse(faces_IndexDel.begin(), faces_IndexDel.end());
	for (int i = 0; i < faces_IndexDel.size(); i++)
	{
		faces_tracking.erase(faces_tracking.begin() + faces_IndexDel.at(i));
		faces_trackingExist.erase(faces_trackingExist.begin() + faces_IndexDel.at(i));
		faces_trackingLastSeen.erase(faces_trackingLastSeen.begin() + faces_IndexDel.at(i));
		faces_trackingStartTime.erase(faces_trackingStartTime.begin() + faces_IndexDel.at(i));
		numberOfFaces++;
	}
	// clear deleting
	faces_IndexDel.clear();

	// -------------------------------------


	// goes through all faces detected
	for (int i = 0; i < faces.size(); i++)
	{
		bool faceExist = false;

		// isit being verify?
		for (int j = 0; j < faces_verifying.size(); j++)
		{
			// make sure all has to be detected again
			// if detected
			if (!(faces[i].x + faces[i].width < faces_verifying[j].x) &&
				!(faces[i].x > faces_verifying[j].x + faces_verifying[j].width) &&
				!(faces[i].y + faces[i].height < faces_verifying[j].y) &&
				!(faces[i].y > faces_verifying[j].y + faces_verifying[j].height)
				)
			{
				faceExist = true;
				break;
			}
		}


		// isit being tracked?
		for (int j = 0; j < faces_tracking.size(); j++)
		{
			if (!(faces[i].x + faces[i].width < faces_tracking[j].x) &&
				!(faces[i].x > faces_tracking[j].x + faces_tracking[j].width) &&
				!(faces[i].y + faces[i].height < faces_tracking[j].y) &&
				!(faces[i].y > faces_tracking[j].y + faces_tracking[j].height)
				)
			{
				faceExist = true;
				break;
			}
		}


		// add to verify if new face
		if (!faceExist)
		{
			cv::Scalar color = cv::Scalar(0, 255, 0);
			rectangle(frame, cvPoint(cvRound(faces.at(i).x*scale), cvRound(faces.at(i).y*scale)), cvPoint(cvRound((faces.at(i).x +
				faces.at(i).width - 1)*scale), cvRound((faces.at(i).y + faces.at(i).height - 1)*scale)), color, 3, 8, 0);
			// must be within distance minDist and maxDist 
			if (distanceValue[(faces.at(i).x + faces.at(i).width / 2) / 4][(faces.at(i).y + faces.at(i).height / 2) / 4] > minDist &&
				distanceValue[(faces.at(i).x + faces.at(i).width / 2) / 4][(faces.at(i).y + faces.at(i).height / 2) / 4] < maxDist)
			{
				// divide by 4 since depth 	
				faces_verifying.push_back(faces[i]); // the roi
				faces_verifyingExist.push_back(true); // existence
				faces_verifyingStartTime.push_back(difftime(timer, mktime(&y2k))); // start time
				faces_verifyingLastSeen.push_back(difftime(timer, mktime(&y2k))); // last seen
			}
		}
	}

	imshow("Detected Face", frame);
}

int main(int argc, char** argv)
{
	astra::initialize();

	set_key_handler();


	// -------------- colour viewer
	sf::RenderWindow windowColour(sf::VideoMode(windowXSize, windowYSize), "Color Viewer");

	astra::StreamSet streamSetColour;
	astra::StreamReader readerColour = streamSetColour.create_reader();

	readerColour.stream<astra::ColorStream>().start();

	ColorFrameListener listenerColour;
	readerColour.add_listener(listenerColour);


	// ------------ depth viewer
	sf::RenderWindow windowDepth(sf::VideoMode(windowXSize, windowYSize), "Depth Viewer");

#ifdef _WIN32
	auto fullscreenStyle = sf::Style::None;
#else
	auto fullscreenStyle = sf::Style::Fullscreen;
#endif

	const sf::VideoMode fullScreenMode = sf::VideoMode::getFullscreenModes()[0];
	const sf::VideoMode windowedMode(windowXSize, windowYSize);
	bool isFullScreen = false;

	astra::StreamSet streamSetDepth;
	astra::StreamReader readerDepth = streamSetDepth.create_reader();
	readerDepth.stream<astra::PointStream>().start();

	auto depthStream = configure_depth(readerDepth);
	depthStream.start();

	DepthFrameListener listenerDepth;

	readerDepth.add_listener(listenerDepth);

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

	astra::terminate();
	return 0;
}
