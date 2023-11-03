#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <xir/graph/graph.hpp>

#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include "vitis/ai/graph_runner.hpp"

#include <chrono>

// fix_point to scale for output tensor
static float get_output_scale(const xir::Tensor* tensor) {
	int fixpos = tensor->template get_attr<int>("fix_point");
	return std::exp2f((float)fixpos);
}

static float get_input_scale(const xir::Tensor* tensor) {
	int fixpos = tensor->template get_attr<int>("fix_point");
	return std::exp2f((float)fixpos);
}

static int preprocess(float i, int input_scale){
	i /= 255;
	i -= 0.5;
	i /= 0.5;
	i *= input_scale;
	return (int)i;
} 

static int postprocess(float i, int output_scale){
	i /= output_scale;
	i += 1;
	i /= 2;
	i = i * 255;
	if(i > 255){
		i = 255;
	}
	if(i < 0){
		i = 0;
	}
	return (int)i;
}


int main(int argc, char* argv[]) {
	//get subgraph of network from xmodel file
	// auto graph = xir::Graph::deserialize(argv[1]);
	// auto root = graph->get_root_subgraph();
	// xir::Subgraph* subgraph = nullptr;
	// for (auto c : root->children_topological_sort()) {
	// 	CHECK(c->has_attr("device"));
	// 	if (c->get_attr<std::string>("device") == "DPU") {
	// 	  subgraph = c;
	// 	  break;
	// 	}
	// }

	// //create external runner
	// //documentation doesn't state the difference beween external and non-external
	// //but I think the external runner automatically allocates io buffers on the CPU.
	// auto attrs = xir::Attrs::create();
	// std::unique_ptr<vart::RunnerExt> runner =
	// vart::RunnerExt::create_runner(subgraph, attrs.get());

	auto graph = xir::Graph::deserialize(argv[1]);
	auto attrs = xir::Attrs::create();
	auto runner =
	  vitis::ai::GraphRunner::create_graph_runner(graph.get(), attrs.get());
	CHECK(runner != nullptr);	


	//these are the functions the normal runner doesn't have
	//it only has get input/output tensor
	auto input_tensor_buffers = runner->get_inputs();
	auto output_tensor_buffers = runner->get_outputs();

	auto input_tensor = input_tensor_buffers[0]->get_tensor();
	auto output_tensor = output_tensor_buffers[0]->get_tensor();



	cv::Mat originalImageIn = cv::imread("test.png");
	cv::Mat imageIn;
	cv::resize(originalImageIn, imageIn, cv::Size(256, 256));

	//raw data address, actually a pointer to array
	uint64_t data_in = 0u;
	//size of array
	size_t size_in = 0u;

	//input_tensor_buffers[0]->data() takes a vector of the same dimension as your tensor
	//and returns the address of that index into the tensor
	//actually returns a tuple so you have to write it into 2 variables
	std::tie(data_in, size_in) = input_tensor_buffers[0]->data(std::vector<int>{0, 0, 0, 0});

	//cast raw address to actual pointer type
	signed char* data_in_char = (signed char*)data_in;
	auto  input_scale = get_input_scale(input_tensor);

	int c = 0;
	for (auto row = 0; row < imageIn.rows; row++) {
		for (auto col = 0; col < imageIn.cols; col++) {
		  auto v = imageIn.at<cv::Vec3b>(row, col);

		  data_in_char[c++] = preprocess(v[2], input_scale);

		  data_in_char[c++] = preprocess(v[1], input_scale);

		  data_in_char[c++] = preprocess(v[0], input_scale);
		}
	}

	//----------------------------------
	//run on FPGA

	for (auto& input : input_tensor_buffers) {
	input->sync_for_write(0, input->get_tensor()->get_data_size() /
							 input->get_tensor()->get_shape()[0]);
	}

	auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
	auto status = runner->wait((int)v.first, -1);


	for (auto& output : output_tensor_buffers) {
	output->sync_for_read(0, output->get_tensor()->get_data_size() /
							  output->get_tensor()->get_shape()[0]);
	}
	//---------------------------------------

	uint64_t data_out = 0u;
	size_t size_out = 0u;

	auto output_scale = get_output_scale(output_tensor);

	std::tie(data_out, size_out) = output_tensor_buffers[0]->data(std::vector<int>{0, 0, 0, 0});

	//need to cast the data_out twice as signed 8 bit data comes out of the network
	//and unsigned 8 bit data is written to the image
	//and I want to use store it all in the same piece of memory
	signed char* data_out_uchar = (signed char*) data_out;
	unsigned char* data_out_char = (unsigned char*) data_out;

	unsigned char r;
	unsigned char b;
	for(int i = 0; i < 253 * 253 * 3; i+=3){
	r = postprocess(data_out_uchar[i+2], output_scale);
	b = postprocess(data_out_uchar[i+0], output_scale);
	data_out_char[i + 0] = r;
	data_out_char[i + 1] = postprocess(data_out_uchar[i+1], output_scale);
	data_out_char[i + 2] = b;
	}

	cv::Mat imageOut = cv::Mat(253, 253, CV_8UC3, data_out_char);
	cv::imwrite("testOut.png", imageOut);
}