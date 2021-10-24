// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 YOLOV5_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// YOLOV5_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
//#ifdef YOLOV5_EXPORTS
//#define YOLOV5_API __declspec(dllexport)
//#else
//#define YOLOV5_API __declspec(dllimport)
//#endif

#ifndef DETECTOR_H
#define DETECTOR_H

#define _CRT_SECURE_NO_WARNINGS

#define DETECT_API __declspec(dllexport)


#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInferRuntime.h>
#include "logging.h"
#include "yololayer.h"



#define USE_FP16  // comment out this if want to use FP32


class DETECT_API Detector

{
public:
	Detector(std::string FileName);
	~Detector();

	bool LoadEngine(std::string FileName);

	std::vector<Yolo::Detection> Detect(cv::Mat& InputMat);
	static Yolo::Detection FindHighestObject(std::vector<Yolo::Detection>& Objects);
	static void DrawRectangle(cv::Mat& InOutMat, std::vector<Yolo::Detection>& Object);
	static void DrawRectangle(cv::Mat& InOutMat, Yolo::Detection& Object);
	float CONF_THRESH = 0.5;

private:
	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;

	static const int DEVICE = 0;  // GPU id
	const float NMS_THRESH = 0.4;
	static const int BATCH_SIZE = 1;

	// stuff we know about the network and the input/output blobs
	int INPUT_H;// = Yolo::INPUT_H;
	int INPUT_W;// = Yolo::INPUT_W;
	int CLASS_NUM;// = Yolo::CLASS_NUM;
	int OUTPUT_SIZE;// = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
	const char* INPUT_BLOB_NAME = "data";
	const char* OUTPUT_BLOB_NAME = "prob";

	Logger gLogger;

	std::string engine_name = "";

	float* data;
	float* prob;


	void* buffers[2];

	cudaStream_t stream;
	int inputIndex;
	int outputIndex;

private:
	void doInference(nvinfer1::IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize);
};

#endif // !DETECTOR_H
