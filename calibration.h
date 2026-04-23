//
// Created by Magnus on 18-04-2026.
//

#pragma once
#include "video.h"
#include "court.h"

bool calibrateCourt(Court& court, VideoSource& video,
    const std::string& saveFile = "calibration.yaml");