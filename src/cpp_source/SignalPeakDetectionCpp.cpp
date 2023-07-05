#include <iostream>
#include <fstream>
#include <limits>

#include <onnxruntime_cxx_api.h>

#include "Signal.h"
#include "IO_utils.h"
#include "Data_utils.h"


int main(int argc, char* argv[])
{
	std::wstring model_path;
	std::string dataset_path;
	int step_factor = 1;

	if (argc < 4)
	{
		std::cout << "Not enough arguments. Program requires:\n";
		std::cout << " - Onnx model path\n";
		std::cout << " - Dataset path where the file info.raw and signal.raw are stored\n";
		std::cout << " - Step factor to downsample the processed signals\n";
		std::cout << "In this order without tags." << std::endl;
		return -1;
	}
	else
	{
		model_path = std::wstring(&argv[1][0], &argv[1][0] + strlen(argv[1]));
		dataset_path = argv[2];
		step_factor = std::atoi(argv[3]);
	}

	std::vector<Signal> signals = IO_utils::read_raw_files(dataset_path);

	Ort::Env env;
	Ort::RunOptions runOptions;
	Ort::Session session(nullptr);

	try
	{
		Ort::SessionOptions ort_session_options;
		session = Ort::Session(env, model_path.c_str(), ort_session_options);
	}
	catch (Ort::Exception& e)
	{
		std::cout << e.what() << std::endl;
		return -1;
	}

	const std::array<int64_t, 3> input_shape = { 1, Signal::signal_length, 1 };
	const std::array<int64_t, 3> output_shape = { 1, Signal::signal_length, 1 };

	std::array<float, Signal::signal_length> input;
	std::array<float, Signal::signal_length> result;

	std::vector<std::array<float, Signal::signal_length>> norm_signals = Data_utils::norm_signals(signals);

	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());
	Ort::Value outputTensor = Ort::Value::CreateTensor<float>(memory_info, result.data(), result.size(), output_shape.data(), output_shape.size());
	
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
	Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
	const std::array<const char*, 1> input_names_array = { input_name.get() };
	const std::array<const char*, 1> output_names_array = { output_name.get() };
	input_name.release();
	output_name.release();


	int TP = 0, TN = 0, FP = 0, FN = 0;

	try 
	{
		std::cout << "Computing predictions." << std::endl;

		for (int i = 0; i < norm_signals.size(); i += step_factor)
		{
			float progress = float(i) / float(norm_signals.size());
			IO_utils::print_progress(progress);

			input = norm_signals[i];
			session.Run(runOptions, input_names_array.data(), &inputTensor, 1, output_names_array.data(), &outputTensor, 1);

			// Thanks to the method used to construct the normalized values array they have the same order of the signal vector
			std::vector<int> identified_peaks;
			for (int j = 0; j < result.size(); j++)
			{
				if (result[j] > 0.5)
				{
					identified_peaks.push_back(j);
					if (signals[i].m_target[j] == 1) TP++;

					else FP++;

				}
				else
				{
					if (signals[i].m_target[j] == 0) TN++;
					
					else FN++;
				}
			}
		}

		std::cout << std::endl;
		std::cout << "Precision:" << float(TP) / float(TP+FP) << std::endl;
		std::cout << "Recall:" << float(TP) / float(TP+FN) << std::endl;

		return 0;
	}
	catch (Ort::Exception& e)
	{
		std::cout << e.what() << std::endl;
		return -1;
	}
}
