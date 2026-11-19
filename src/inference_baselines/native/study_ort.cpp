// Generic static-shape ORT CPU adapter. No shape-specific kernels.
#include "onnxruntime_cxx_api.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
namespace {
thread_local std::string error;
struct Runner {
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "inference-study"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::vector<float> input;
    std::array<float,3> output{};
    std::unique_ptr<Ort::Value> in, out;
    std::unique_ptr<Ort::IoBinding> binding;
    Ort::RunOptions run{nullptr};
    Runner(const char* path, int64_t history, int64_t features) {
        options.SetIntraOpNumThreads(1); options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session=std::make_unique<Ort::Session>(env,path,options);
        auto input_info=session->GetInputTypeInfo(0);
        auto output_info=session->GetOutputTypeInfo(0);
        if(session->GetInputCount()!=1 || session->GetOutputCount()!=1 ||
           input_info.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,history,features} ||
           output_info.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,3} ||
           input_info.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           output_info.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Expected static FP32 [1,history,features] -> [1,3]");
        input.resize(history*features);
        std::array<int64_t,3> ishape{1,history,features}; std::array<int64_t,2> oshape{1,3};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        in=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory,input.data(),input.size(),ishape.data(),3));
        out=std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(memory,output.data(),3,oshape.data(),2));
        binding=std::make_unique<Ort::IoBinding>(*session);
        binding->BindInput("input",*in); binding->BindOutput("logits",*out);
    }
};
}
extern "C" {
const char* study_error(){return error.c_str();}
const char* study_ort_version(){return OrtGetApiBase()->GetVersionString();}
void* study_create(const char* path,int64_t history,int64_t features) {
    try {return new Runner(path,history,features);} catch(const std::exception& e){error=e.what();return nullptr;}
}
void study_destroy(void* pointer){delete static_cast<Runner*>(pointer);}
double study_call(void* pointer,const float* source,float* output,int prepared,int* decision){
    try {
        auto& r=*static_cast<Runner*>(pointer);
        if(prepared) std::memcpy(r.input.data(),source,r.input.size()*sizeof(float));
        auto start=std::chrono::steady_clock::now();
        if(!prepared) std::memcpy(r.input.data(),source,r.input.size()*sizeof(float));
        r.session->Run(r.run,*r.binding);
        int chosen=static_cast<int>(std::max_element(r.output.begin(),r.output.end())-r.output.begin());
        *decision=chosen;
        auto end=std::chrono::steady_clock::now();
        std::memcpy(output,r.output.data(),3*sizeof(float)); // audit retention outside timer
        return std::chrono::duration<double,std::nano>(end-start).count();
    } catch(const std::exception& e){error=e.what();return -1;}
}
}
