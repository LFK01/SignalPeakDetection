#pragma once

#include <vector>
#include <numeric>
#include <algorithm>

#include "Signal.h"


class Data_utils
{
public:
	static std::vector<std::array<float, Signal::signal_length>> norm_signals(const std::vector<Signal> &signals)
	{
		std::vector<std::array<float, Signal::signal_length>> normalized_signals;

		int max_value = 0;
		for (Signal signal : signals)
		{
			int signal_max = *std::max_element(signal.m_data, signal.m_data + Signal::signal_length);
			if (signal_max > max_value)
			{
				max_value = signal_max;
			}
		}

		float mean_value = 0.0;
		int sum = 0;
		for (Signal signal : signals)
		{
			sum += std::reduce(signal.m_data, signal.m_data + Signal::signal_length);
		}

		mean_value = float(sum) / (signals.size() * 1024);

		for (int i = 0; i < signals.size(); i++)
		{
			std::array<float, Signal::signal_length> norm_values;
			for (int j = 0; j < Signal::signal_length; j++)
			{
				// min value is 0
				norm_values[j] = (signals[i].m_data[j] - mean_value) / max_value;
			}

			normalized_signals.push_back(norm_values);
		}

		return normalized_signals;
	};
};
