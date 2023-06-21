#pragma once

#include <vector>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>
#include "Signal.h"

class IO_utils
{
public:
	static std::vector<Signal> read_raw_files(const std::string &dataset_path) 
	{

		std::string info_filename = dataset_path + "\\info.raw";
		std::string signal_filename = dataset_path + "\\signal.raw";

		std::ifstream fin(info_filename, std::ios::binary | std::ios::in);

		if (!fin)
		{
			std::cout << "info.raw file not found!" << std::endl;
		}

		std::vector<float> info_vector;

		float float_iterator;
		while (fin.read((char*)&float_iterator, sizeof(float_iterator))) {
			info_vector.insert(info_vector.end(), float_iterator);
		}

		std::cout << "Read Info file." << std::endl;

		fin = std::ifstream(signal_filename, std::ios::binary);

		if (!fin)
		{
			std::cout << "signal.raw file not found!" << std::endl;
		}

		std::vector<uint8_t> signal_vector;

		uint8_t uint_iterator;
		while (fin.read((char*)&uint_iterator, sizeof(uint_iterator))) {
			signal_vector.push_back(uint_iterator);
		}

		std::cout << "Read Signal file." << std::endl;

		std::vector<Signal> signals;

		std::cout << "Building signals objects:" << std::endl;

		int signal_index = 0;
		while (signal_index * 11 < info_vector.size()) {
			float progress = float(signal_index*11) / float(info_vector.size());

			if (signal_index % 100 == 0)
			{
				IO_utils::print_progress(progress);
			}

			int info_values_index = signal_index * 11;
			float signal_length = info_vector[info_values_index];
			float peak_number = info_vector[info_values_index + 1];

			std::vector<float> peaks;
			for (int i = 0; i<int(peak_number); i++) {
				peaks.push_back(info_vector[info_values_index + 2 + i * 3]);
			}

			std::array<int, Signal::signal_length> data;
			std::copy(signal_vector.begin() + signal_index * 1024, signal_vector.begin() + (signal_index + 1) * 1024, data.begin());

			signals.push_back(Signal(peaks, data));
			signal_index++;
		}

		std::cout << "\nDone building signals." << std::endl;

		return signals;
	};

	static void clrscr()
	{
		std::cout << "\033[2J\033[1;1H";
	};

	static void print_progress(float progress)
	{
		int barWidth = 50;

		std::cout << "[";
		int pos = barWidth * progress;
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << int(progress * 100.0) << " %\r";
		std::cout.flush();
	};
};

