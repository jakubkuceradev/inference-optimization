// Native batch-one APIs; declarations follow the versions cited in tree288.json/README.
#include <dlfcn.h>
#include <sched.h>
#include <sys/resource.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Row = std::array<float, 288>;
using Scores = std::array<double, 3>;

struct Library {
    void* lib;
    explicit Library(const std::string& path) : lib(dlopen(path.c_str(), RTLD_NOW|RTLD_LOCAL)) {
        if (!lib) throw std::runtime_error(dlerror());
    }
    template<class T> T symbol(const char* name) {
        void* p = dlsym(lib, name);
        if (!p) throw std::runtime_error(name);
        return reinterpret_cast<T>(p);
    }
    ~Library() { dlclose(lib); }
};

struct Engine {
    Scores output{};
    virtual const Scores& predict() = 0;
    virtual ~Engine() = default;
};

struct XGBoost : Engine {
    Library library;
    void* handle{};
    const char* (*error)();
    int (*free_model)(void*);
    int (*invoke)(void*,const char*,const char*,void*,const uint64_t**,uint64_t*,const float**);
    std::string array;
    const std::string config = "{\"type\":0,\"training\":false,\"iteration_begin\":0,\"iteration_end\":0,\"missing\":NaN,\"strict_shape\":true,\"cache_id\":0}";
    void check(int status) { if (status) throw std::runtime_error(error()); }
    XGBoost(const std::string& lib, const std::string& model, Row& row) : library(lib) {
        error = library.symbol<const char*(*)()>("XGBGetLastError");
        auto create = library.symbol<int(*)(void* const*,uint64_t,void**)>("XGBoosterCreate");
        auto load = library.symbol<int(*)(void*,const char*)>("XGBoosterLoadModel");
        auto param = library.symbol<int(*)(void*,const char*,const char*)>("XGBoosterSetParam");
        free_model = library.symbol<int(*)(void*)>("XGBoosterFree");
        invoke = library.symbol<decltype(invoke)>("XGBoosterPredictFromDense");
        check(create(nullptr,0,&handle)); check(load(handle,model.c_str()));
        check(param(handle,"nthread","1")); check(param(handle,"device","cpu"));
        array = "{\"data\":["+std::to_string(reinterpret_cast<uintptr_t>(row.data()))+",false],\"shape\":[1,288],\"typestr\":\"<f4\",\"version\":3}";
    }
    const Scores& predict() override {
        const uint64_t* shape{}; uint64_t dims{}; const float* values{};
        check(invoke(handle,array.c_str(),config.c_str(),nullptr,&shape,&dims,&values));
        if (dims!=2 || shape[0]!=1 || shape[1]!=3) throw std::runtime_error("XGBoost output shape");
        std::copy_n(values,3,output.begin()); return output;
    }
    ~XGBoost() { if (handle) free_model(handle); }
};

struct LightGBM : Engine {
    Library library;
    void* handle{}; void* fast{};
    const float* row;
    const char* (*error)();
    int (*free_model)(void*); int (*free_fast)(void*);
    int (*invoke)(void*,const void*,int64_t*,double*);
    void check(int status) { if (status) throw std::runtime_error(error()); }
    LightGBM(const std::string& lib,const std::string& model,Row& input) : library(lib),row(input.data()) {
        error = library.symbol<const char*(*)()>("LGBM_GetLastError");
        auto load = library.symbol<int(*)(const char*,int*,void**)>("LGBM_BoosterCreateFromModelfile");
        auto init = library.symbol<int(*)(void*,int,int,int,int,int32_t,const char*,void**)>("LGBM_BoosterPredictForMatSingleRowFastInit");
        free_model = library.symbol<int(*)(void*)>("LGBM_BoosterFree");
        free_fast = library.symbol<int(*)(void*)>("LGBM_FastConfigFree");
        invoke = library.symbol<decltype(invoke)>("LGBM_BoosterPredictForMatSingleRowFast");
        int iterations{};
        check(load(model.c_str(),&iterations,&handle));
        check(init(handle,0,0,-1,0,288,"num_threads=1",&fast));
    }
    const Scores& predict() override {
        int64_t length{}; check(invoke(fast,row,&length,output.data()));
        if (length!=3) throw std::runtime_error("LightGBM output shape");
        return output;
    }
    ~LightGBM() { if(fast) free_fast(fast); if(handle) free_model(handle); }
};

struct CatBoost : Engine {
    Library library;
    void* handle{};
    const float* rows[1];
    const char* (*error)();
    void (*free_model)(void*);
    bool (*invoke)(void*,size_t,const float**,size_t,double*,size_t);
    void check(bool ok) { if(!ok) throw std::runtime_error(error()); }
    CatBoost(const std::string& lib,const std::string& model,Row& input) : library(lib),rows{input.data()} {
        error = library.symbol<const char*(*)()>("GetErrorString");
        auto create = library.symbol<void*(*)()>("ModelCalcerCreate");
        auto load = library.symbol<bool(*)(void*,const char*)>("LoadFullModelFromFile");
        auto type = library.symbol<bool(*)(void*,int)>("SetPredictionType");
        free_model = library.symbol<void(*)(void*)>("ModelCalcerDelete");
        invoke = library.symbol<decltype(invoke)>("CalcModelPredictionFlat");
        handle=create(); check(handle!=nullptr); check(load(handle,model.c_str()));
        check(type(handle,3)); // APT_PROBABILITY, CatBoost 1.2.10 C API.
    }
    const Scores& predict() override {
        check(invoke(handle,1,rows,288,output.data(),3)); return output;
    }
    ~CatBoost() { if(handle) free_model(handle); }
};

std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> result; std::istringstream in(line); std::string item;
    while(std::getline(in,item,'\t')) result.push_back(item);
    return result;
}

template<class T> std::vector<T> read(const std::string& path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in) throw std::runtime_error("Cannot read "+path);
    const auto bytes=in.tellg();
    if(bytes<=0 || bytes%sizeof(T)) throw std::runtime_error("Invalid array "+path);
    std::vector<T> result(static_cast<size_t>(bytes)/sizeof(T)); in.seekg(0);
    if(!in.read(reinterpret_cast<char*>(result.data()),bytes)) throw std::runtime_error("Short read");
    return result;
}

template<class T> void write(const std::string& path,const std::vector<T>& values) {
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
    if(!out) throw std::runtime_error("Cannot write "+path);
}

struct Rolling {
    std::array<std::array<float,144>,5> ring{};
    size_t oldest{};
    void reset(const float* prefill) {
        for(size_t j=0;j<5;++j) std::copy_n(prefill+j*144,144,ring[j].begin());
        oldest=0;
    }
    void update(const float* incoming,Row& output) {
        std::copy_n(incoming,144,ring[oldest].begin());
        std::copy_n(incoming,144,output.begin());
        oldest=(oldest+1)%5;
        const auto& a=ring[oldest]; const auto& b=ring[(oldest+1)%5];
        const auto& c=ring[(oldest+2)%5]; const auto& d=ring[(oldest+3)%5];
        const auto& e=ring[(oldest+4)%5];
        // Fixed five-row window, ordered FP64 sum: matches the training recipe.
        for(size_t f=0;f<144;++f) {
            double sum=0.; sum+=a[f]; sum+=b[f]; sum+=c[f]; sum+=d[f]; sum+=e[f];
            output[144+f]=static_cast<float>(sum/5.);
        }
    }
};

void selftest() {
    std::vector<float> sequence(110*144);
    for(size_t i=0;i<sequence.size();++i) sequence[i]=static_cast<float>((static_cast<int>(i)%97)-48)/7.f;
    Rolling rolling; rolling.reset(sequence.data()); Row row{};
    for(size_t t=5;t<110;++t) {
        rolling.update(sequence.data()+t*144,row);
        for(size_t f=0;f<144;++f) {
            double sum=0.; for(size_t j=t-4;j<=t;++j) sum+=sequence[j*144+f];
            if(row[f]!=sequence[t*144+f] || row[f+144]!=static_cast<float>(sum/5.))
                throw std::runtime_error("Rolling feature selftest failed");
        }
    }
}

int main(int argc,char** argv) try {
    if(argc==2 && std::string(argv[1])=="selftest") { selftest(); return 0; }
    if(argc!=6) throw std::runtime_error("Expected predict manifest features output cpu OR benchmark manifest output cpu warmup");
    const std::string mode=argv[1],manifest=argv[2];
    int cpu=std::stoi(mode=="predict"?argv[5]:argv[4]);
    if(cpu<0 || cpu>=CPU_SETSIZE) throw std::runtime_error("Invalid CPU");
    cpu_set_t affinity; CPU_ZERO(&affinity); CPU_SET(cpu,&affinity);
    if(sched_setaffinity(0,sizeof(affinity),&affinity)) throw std::runtime_error("CPU affinity failed");
    Row row{}; std::map<std::string,std::unique_ptr<Engine>> engines;
    std::ifstream list(manifest); if(!list) throw std::runtime_error("Missing manifest");
    std::string line;
    while(std::getline(list,line)) {
        const auto f=fields(line); if(f.size()!=4) throw std::runtime_error("Invalid model manifest");
        if(f[1]=="xgboost") engines[f[0]]=std::make_unique<XGBoost>(f[2],f[3],row);
        else if(f[1]=="lightgbm") engines[f[0]]=std::make_unique<LightGBM>(f[2],f[3],row);
        else if(f[1]=="catboost") engines[f[0]]=std::make_unique<CatBoost>(f[2],f[3],row);
        else throw std::runtime_error("Unknown engine");
    }
    if(engines.empty()) throw std::runtime_error("Empty model manifest");
    if(mode=="predict") {
        const auto input=read<float>(argv[3]);
        if(input.size()%288) throw std::runtime_error("Invalid feature dimensions");
        const size_t count=input.size()/288;
        for(auto& [name,engine]:engines) {
            std::vector<double> values(count*3);
            for(size_t i=0;i<count;++i) {
                std::copy_n(input.data()+i*288,288,row.begin());
                const auto& value=engine->predict(); std::copy_n(value.begin(),3,values.begin()+i*3);
            }
            write(std::string(argv[4])+"/"+name+"-probabilities.f64",values);
            std::cout<<"native quality "<<name<<" "<<count<<"\n"<<std::flush;
        }
        return 0;
    }
    if(mode!="benchmark") throw std::runtime_error("Unknown mode");
    const std::string root=argv[3];
    const auto incoming=read<float>(root+"/snapshots.f32");
    const auto prepared=read<float>(root+"/prepared.f32");
    const auto prefill=read<float>(root+"/prefill.f32");
    const size_t count=incoming.size()/144;
    if(incoming.size()%144 || prepared.size()!=count*288 || prefill.size()!=5*144) throw std::runtime_error("Invalid stream shapes");
    Rolling rolling; rolling.reset(prefill.data());
    for(size_t i=0;i<count;++i) {
        rolling.update(incoming.data()+i*144,row);
        if(!std::equal(row.begin(),row.end(),prepared.begin()+i*288)) throw std::runtime_error("Streaming/prepared feature mismatch");
    }
    for(auto& [name,engine]:engines) for(int i=0;i<std::stoi(argv[5]);++i) engine->predict();
    std::ifstream schedule(root+"/schedule.tsv"); if(!schedule) throw std::runtime_error("Missing schedule");
    volatile int consumed=0;
    while(std::getline(schedule,line)) {
        const auto f=fields(line); if(f.size()!=3) throw std::runtime_error("Invalid schedule");
        auto& engine=engines.at(f[1]);
        if(f[2]!="prepared" && f[2]!="local_preparation") throw std::runtime_error("Unknown boundary");
        rolling.reset(prefill.data());
        std::vector<int64_t> ns(count); std::vector<double> values(count*3);
        for(size_t i=0;i<count;++i) {
            if(f[2]=="prepared") std::copy_n(prepared.data()+i*288,288,row.begin());
            const auto before=std::chrono::steady_clock::now();
            if(f[2]=="local_preparation") rolling.update(incoming.data()+i*144,row);
            const auto& value=engine->predict();
            consumed=std::max_element(value.begin(),value.end())-value.begin();
            const auto after=std::chrono::steady_clock::now();
            ns[i]=std::chrono::duration_cast<std::chrono::nanoseconds>(after-before).count();
            std::copy_n(value.begin(),3,values.begin()+i*3);
        }
        const std::string stem=root+"/"+f[1]+"-"+f[2]+"-"+f[0];
        write(stem+"-ns.i64",ns); write(stem+"-probabilities.f64",values);
        std::cout<<"timed "<<f[1]<<" "<<f[2]<<" repeat"<<f[0]<<"\n"<<std::flush;
    }
    rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::ofstream metadata(root+"/native-metadata.json");
    metadata<<"{\"peak_process_rss_kib\":"<<usage.ru_maxrss<<",\"cpu\":"<<cpu
            <<",\"model_count\":"<<engines.size()<<",\"consumed_class\":"<<consumed
            <<",\"rolling_feature_check\":\"all streamed FP32 features equal prepared features\"}\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<"\n"; return 1;
}
