//
// Created by Magnus on 18-04-2026.
//

#include "calibration.h"

struct CalibrationState {
    std::vector<cv::Point2f> clickedPoints;
    int currentIndex = 0;
};

static constexpr const char* WINDOW_NAME = "Video";

void onMouse(int event, int x, int y, int flags, void* userdata) {
    auto* state = static_cast<CalibrationState*>(userdata);

    if (event != cv::EVENT_LBUTTONDOWN) return;
    if (state->currentIndex >= Court::landmarkWorldPositions().size()) return;

    state->clickedPoints.push_back(cv::Point2f(x, y));
    state->currentIndex++;
}

bool calibrateCourt(Court& court, VideoSource& video,
    const std::string& saveFile) {
    CalibrationState state;
    if (cv::FileStorage fs("calibration.yaml", cv::FileStorage::READ); fs.isOpened()) {
        fs["imagePoints"] >> state.clickedPoints;
        fs.release();
        state.currentIndex = static_cast<int>(state.clickedPoints.size());
    }
    const auto landmarks = Court::landmarkWorldPositions();
    const auto names = Court::landmarkNames();
    const int totalLandmarks = static_cast<int>(landmarks.size());

    cv::namedWindow(WINDOW_NAME);
    cv::setMouseCallback("Video", onMouse, &state);
    int lastIndex = -1;

    while (state.currentIndex < totalLandmarks) {
        cv::Mat courtImage = Court::drawWithPoints(landmarks, state.currentIndex);
        cv::imshow("Court", courtImage);
        cv::imshow("Video", video.getFrame());

        if (state.currentIndex != lastIndex) {
            std::cout << "Click: " << names[state.currentIndex] << std::endl;
            std::cout << " (" << state.currentIndex + 1 << "/" << totalLandmarks << ")" << std::endl;
            lastIndex = state.currentIndex;
        }

        int key = cv::waitKey(30);
        if (key == 'q') break;
        if (key == 'n') {
            video.nextFrame();
        }
        else if (key == 'p') {
            video.prevFrame();
        }
    }
    if (state.clickedPoints.size() != totalLandmarks) {
        return false;
    }

    if (!court.computeHomography(state.clickedPoints, landmarks)) {
        return false;
    }

    cv::FileStorage writeFs(saveFile, cv::FileStorage::WRITE);
    writeFs << "imagePoints" << state.clickedPoints;
    writeFs.release();

    return true;






}
