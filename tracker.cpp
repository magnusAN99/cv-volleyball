//
// Created by Magnus on 24-04-2026.
//

#include "tracker.h"

void Tracker::update(const std::vector<cv::Point2f>& detections) {
    //Track matching logic
    std::vector detectionUsed(detections.size(), false);

    for (auto& track : tracks_) {
        int bestIdx = -1;
        float bestDist = maxMatchDistance_;

        for (size_t i = 0; i < detections.size(); i++) {
            if (detectionUsed[i]) continue;

            const float dx = detections[i].x - track.worldPosition.x;
            const float dy = detections[i].y - track.worldPosition.y;

            if (const float dist = std::sqrt(dx * dx + dy * dy); dist < bestDist) {
                bestDist = dist;
                bestIdx = static_cast<int>(i);
            }
        }

        if (bestIdx >= 0) {
            track.worldPosition.x = (1 - alpha_) * track.worldPosition.x + alpha_ * detections[bestIdx].x;
            track.worldPosition.y = (1 - alpha_) * detections[bestIdx].y + alpha_ * detections[bestIdx].y;

            track.framesSinceLastSeen = 0;
            detectionUsed[bestIdx] = true;
        } else {
            track.framesSinceLastSeen++;
        }
    }

    for (size_t i = 0; i < detections.size(); i++) {
        if (detectionUsed[i]) continue;
        tracks_.push_back( {nextId_++, detections[i], 0} );
    }

    std::erase_if(tracks_,
                  [this](const Track& t) { return t.framesSinceLastSeen > maxStaleFrames_; });

    std::vector<cv::Point2f> trackPoints;
    for (const auto& track : tracks_) {
        trackPoints.push_back(track.worldPosition);
    }
}

std::vector<cv::Point2f> Tracker::getActivePositions(int minAge) const {
    std::vector<cv::Point2f> activePositions;
    for (const auto& track : tracks_) {
        if (track.framesSinceLastSeen == 0) {
            activePositions.push_back(track.worldPosition);
        }
    }
    return activePositions;
}

