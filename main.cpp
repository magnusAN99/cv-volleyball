#include <iostream>
#include "video.h"
#include "court.h"
#include "calibration.h"
#include "detector.h"

int main() {
    struct Track {
        int id;
        cv::Point2f worldPosition;
        int framesSinceLastSeen;
    };
    std::vector<Track> tracks;
    int nextTrackId = 0;

    VideoSource video;

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

        //Track matching logic
        std::vector<bool> detectionUsed(playerPoints.size(), false);

        for (auto& track : tracks) {
            constexpr float maxMatchDistance = 1.5f;
            int bestIdx = -1;
            float bestDist = maxMatchDistance;

            for (size_t i = 0; i < playerPoints.size(); i++) {
                if (detectionUsed[i]) continue;

                float dx = playerPoints[i].x - track.worldPosition.x;
                float dy = playerPoints[i].y - track.worldPosition.y;
                float dist = std::sqrt(dx * dx + dy * dy);

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = static_cast<int>(i);
                }
            }

            if (bestIdx >= 0) {
                constexpr float alpha = 0.3f;
                track.worldPosition.x = (1 - alpha) * track.worldPosition.x + alpha * playerPoints[bestIdx].x;
                track.worldPosition.y = (1 - alpha) * playerPoints[bestIdx].y + alpha * playerPoints[bestIdx].y;

                track.framesSinceLastSeen = 0;
                detectionUsed[bestIdx] = true;
            } else {
                track.framesSinceLastSeen++;
            }
        }

        for (size_t i = 0; i < playerPoints.size(); i++) {
            if (detectionUsed[i]) continue;
            tracks.push_back( {nextTrackId++, playerPoints[i], 0} );
        }

        constexpr int maxStaleFrames = 5;
        std::erase_if(tracks,
                      [](const Track& t) { return t.framesSinceLastSeen > maxStaleFrames; });

        std::vector<cv::Point2f> trackPoints;
        for (const auto& track : tracks) {
            trackPoints.push_back(track.worldPosition);
        }

        cv::imshow("Video", annotated);
        cv::imshow("Minimap", court.drawWithPlayers(trackPoints));

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