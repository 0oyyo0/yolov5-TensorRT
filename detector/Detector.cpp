#include "Detector.h"
#include <iostream>
#include "common.hpp"
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"

using namespace nvinfer1;


Detector::Detector(std::string FileName)
{

    INPUT_H = Yolo::INPUT_H;
    INPUT_W = Yolo::INPUT_W;
    CLASS_NUM = Yolo::CLASS_NUM;
    OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    this->data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    this->prob = new float[BATCH_SIZE * OUTPUT_SIZE];
    //在.h中定义的，后期改为选择性能更好的GPU 
    cudaSetDevice(DEVICE);
    this->engine_name = FileName;
    std::ifstream file(engine_name, std::ios::binary);
    //std::ifstream file("C:\\Users\\yangy\\Desktop\\yolov5\\build\\Debug\\best.engine", std::ios::binary);

    if (!file.good()) { 
        std::cerr << "read " << engine_name << " error!" << std::endl;
        throw "fail to read weight file";
        return;
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();


    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);
    delete[] trtModelStream;
    assert(this->engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = this->engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = this->engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream));

}

Detector::~Detector()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();

    delete[]this->data;
    delete[]this->prob;
}

bool Detector::LoadEngine(std::string FileName)
{
    return true;
}

std::vector<Yolo::Detection> Detector::Detect(cv::Mat& InputMat)
{
    //auto start = std::chrono::system_clock::now();
    //this->ProcessImage(InputMat);
    cv::Mat pr_img = preprocess_img(InputMat, INPUT_W, INPUT_H); // letterbox BGR to RGB
    //cv::Mat pr_img = InputMat;

    int i = 0;
    int offset = INPUT_H * INPUT_W;
    int offset2 = offset * 2;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + offset] = (float)uc_pixel[1] / 255.0;
            data[i + offset2] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    // Run inference

    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    //auto end = std::chrono::system_clock::now();
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<Yolo::Detection> batch_res;

    nms(batch_res, &prob[0], CONF_THRESH, NMS_THRESH);

    return batch_res;
}

Yolo::Detection Detector::FindHighestObject(std::vector<Yolo::Detection>& Objects)
{
    if (Objects.size() == 0)
        return Yolo::Detection();
    int index = 0;
    float HighestValue = 0;

    for (size_t i = 0; i < Objects.size(); i++)
    {
        if (Objects[i].conf > HighestValue)
        {
            HighestValue = Objects[i].conf;
            index = i;
        }
    }

    return Objects[index];
}

void Detector::DrawRectangle(cv::Mat& InOutMat, std::vector<Yolo::Detection>& Object)
{
    for (size_t j = 0; j < Object.size(); j++) {
        cv::Rect r = get_rect(InOutMat, Object[j].bbox);
        cv::rectangle(InOutMat, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(InOutMat, std::to_string((int)Object[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}

void Detector::DrawRectangle(cv::Mat& InOutMat, Yolo::Detection& Object)
{
    cv::Rect r = get_rect(InOutMat, Object.bbox);
    cv::rectangle(InOutMat, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    cv::putText(InOutMat, std::to_string((int)(Object.conf * 100)), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0x00, 0x00, 0xFF), 2);
}

void Detector::doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}










