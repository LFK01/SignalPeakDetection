#include "Signal.h"
#include <initializer_list>

#define MAX_PEAK_NUMBER 3
#define SIGNAL_LENGTH 1024


Signal::Signal(std::vector<float>& peaks, std::array<int, SIGNAL_LENGTH>& data)
{
    m_peaks = peaks;
    std::copy(std::begin(data), std::end(data), std::begin(m_data));

    std::fill(std::begin(m_target), std::end(m_target), 0);

    for (float peak : peaks) {
        m_target[int(peak)] = 1;
    }
};