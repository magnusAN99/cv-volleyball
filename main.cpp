#include <iostream>
#include "video.h"
#include "court.h"
#include "calibration.h"
#include "detector.h"
#include "tracker.h"

int main() {
    VideoSource video;
    Tracker tracker;

    if (!video.open("D:/hygge/videos/s2.mp4")) {
        std::cerr << "Could not open video file" << std::endl;
        return 1;
    }

    Court court;
    if (!calibrateCourt(court, video)) {
        std::cerr << "Could not calibrate court" << std::endl;
        return 1;
    }
    Detector detector("models/yolov8n.onnx");
    bool playing = false;

    while (true) {
        cv::Mat frame = video.getFrame();
        std::vector<cv::Point2f> playerPoints;
        auto detections = detector.detect(frame);
        cv::Mat annotated = frame.clone();
        for (const auto& detc : detections) {
            cv::Point2f footPosition = {detc.box.x + detc.box.width / 2.0f,
                detc.box.y + detc.box.height / 1.0f};
            if (court.isOnCourt(footPosition, 3.0f)) {
                cv::rectangle(annotated, detc.box, cv::Scalar(0, 255, 0), 2);
                playerPoints.push_back(court.imageToWorld(footPosition));
            }
        }

        tracker.update(playerPoints);

        cv::imshow("Video", annotated);
        cv::imshow("Minimap", Court::drawWithPlayers(tracker.getActivePositions()));

        int key = cv::waitKey(playing ? 1 : 0);
        if (key == 'q') break;
        if (key == ' ') playing = !playing;
        else if (key == 'n') video.nextFrame();
        else if (key == 'p') video.prevFrame();
        else if (key == 'f') video.setFrame(video.currentFrameNumber() + 300);
        else if (key == 'r') video.setFrame(video.currentFrameNumber() - 300);

        if (playing) video.nextFrame();
    }

    return 0;
}