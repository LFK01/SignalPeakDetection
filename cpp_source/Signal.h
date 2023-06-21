#pragma once


#include <array>
#include <vector>

#define MAX_PEAK_NUMBER 3
#define SIGNAL_LENGTH 1024

class Signal
{
public:
    static const int signal_length = SIGNAL_LENGTH;
    static const int max_peak_number = MAX_PEAK_NUMBER;

    Signal(std::vector<float>& peaks, std::array<int, SIGNAL_LENGTH>& data);

    std::vector<float> m_peaks;
    int m_data[SIGNAL_LENGTH];
    int m_target[SIGNAL_LENGTH];
};