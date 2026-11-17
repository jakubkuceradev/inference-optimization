#include "features.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Mapped {
    int fd; size_t bytes; void* address;
    explicit Mapped(const char* path) {
        fd=open(path,O_RDONLY); if(fd<0) throw std::runtime_error("open input");
        struct stat st{}; if(fstat(fd,&st)) throw std::runtime_error("stat input");
        bytes=st.st_size; address=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
        if(address==MAP_FAILED) throw std::runtime_error("map input");
    }
    ~Mapped(){munmap(address,bytes);close(fd);}
    const float* floats(size_t offset=0) const {return reinterpret_cast<const float*>(static_cast<const char*>(address)+offset);}
};
class Predictor {
    void* library=nullptr; void* model=nullptr;
    bool (*calculate)(void*,size_t,const float**,size_t,double*,size_t)=nullptr;
    void (*destroy)(void*)=nullptr;
    std::vector<double> weights;
    size_t dimension;
public:
    Predictor(const std::string& kind,const char* path,const char* lib,size_t d):dimension(d) {
        if(kind=="ridge") {
            weights.resize(3*d+3); std::ifstream in(path,std::ios::binary);
            in.read(reinterpret_cast<char*>(weights.data()),weights.size()*8);
            if(!in || in.peek()!=EOF) throw std::runtime_error("ridge weight length");
        } else {
            library=dlopen(lib,RTLD_NOW|RTLD_LOCAL); if(!library) throw std::runtime_error(dlerror());
            auto create=reinterpret_cast<void*(*)()>(dlsym(library,"ModelCalcerCreate"));
            auto load=reinterpret_cast<bool(*)(void*,const char*)>(dlsym(library,"LoadFullModelFromFile"));
            auto type=reinterpret_cast<bool(*)(void*,int)>(dlsym(library,"SetPredictionType"));
            calculate=reinterpret_cast<decltype(calculate)>(dlsym(library,"CalcModelPredictionFlat"));
            destroy=reinterpret_cast<decltype(destroy)>(dlsym(library,"ModelCalcerDelete"));
            if(!create||!load||!type||!calculate||!destroy) throw std::runtime_error("CatBoost symbols");
            model=create(); if(!model||!load(model,path)||!type(model,3)) throw std::runtime_error("CatBoost load");
        }
    }
    ~Predictor(){if(model)destroy(model);if(library)dlclose(library);}
    void predict(const float* row,double* out) {
        if(model) { if(!calculate(model,1,&row,dimension,out,3)) throw std::runtime_error("CatBoost prediction"); }
        else {
            std::copy_n(weights.data()+3*dimension,3,out);
            for(size_t f=0;f<dimension;++f) for(size_t c=0;c<3;++c) out[c]+=static_cast<double>(row[f])*weights[3*f+c];
        }
    }
};
int main(int argc,char** argv) try {
    if(argc<2) throw std::runtime_error("missing mode");
    std::string mode=argv[1];
    if(mode=="features") {
        if(argc!=9) throw std::runtime_error("features recipe npy offset first stop output append");
        Features features(argv[2]); Mapped source(argv[3]); size_t offset=std::stoull(argv[4]);
        size_t first=std::stoull(argv[5]),stop=std::stoull(argv[6]);
        if(first+1<features.history() || stop<=first || offset+stop*144*4>source.bytes) throw std::runtime_error("feature range");
        const float* raw=source.floats(offset); std::vector<float> row(features.dimension());
        features.reset(raw+(first+1-features.history())*144);
        std::ofstream out(argv[7],std::ios::binary|(std::string(argv[8])=="1"?std::ios::app:std::ios::trunc));
        for(size_t e=first;e<stop;++e) {
            if(e==first) features.emit(row.data()); else features.update(raw+e*144,row.data());
            out.write(reinterpret_cast<const char*>(row.data()),row.size()*4);
        }
        if(!out) throw std::runtime_error("feature write");
    } else {
        // run recipe kind model library features.f32 output mode snapshots.f32 prefill.f32 count
        if(argc!=12) throw std::runtime_error("run argument count");
        Features features(argv[2]); size_t d=features.dimension();
        Predictor predictor(argv[3],argv[4],argv[5],d); Mapped rows(argv[6]);
        size_t count=std::stoull(argv[11]); if(count*d*4>rows.bytes) throw std::runtime_error("prediction range");
        bool bench=std::string(argv[8])!="predict",stream=std::string(argv[8])=="stream";
        std::vector<float> row(d); double out[3];
        std::ofstream output(argv[7],std::ios::binary);
        if(!bench) {
            for(size_t i=0;i<count;++i) {predictor.predict(rows.floats()+i*d,out);output.write(reinterpret_cast<char*>(out),24);}
        } else {
            Mapped snapshots(argv[9]),prefill(argv[10]);
            if(snapshots.bytes<count*144*4 || prefill.bytes!=384*144*4) throw std::runtime_error("stream input size");
            features.reset(prefill.floats()+(384-features.history())*144);
            for(size_t i=0;i<count;++i) {
                features.update(snapshots.floats()+i*144,row.data());
                if(std::memcmp(row.data(),rows.floats()+i*d,d*4)) throw std::runtime_error("offline/online feature mismatch");
            }
            for(size_t i=0;i<100;++i) predictor.predict(rows.floats()+(i%count)*d,out);
            features.reset(prefill.floats()+(384-features.history())*144);
            std::vector<double> records(count*5); volatile int consumed=0;
            for(size_t i=0;i<count;++i) {
                if(!stream) std::copy_n(rows.floats()+i*d,d,row.data());
                auto start=std::chrono::steady_clock::now();
                if(stream) features.update(snapshots.floats()+i*144,row.data());
                predictor.predict(row.data(),out);
                int decision=out[1]>out[0]?1:0; if(out[2]>out[decision])decision=2; consumed=decision;
                auto end=std::chrono::steady_clock::now();
                records[5*i]=std::chrono::duration<double,std::nano>(end-start).count();
                std::copy_n(out,3,records.data()+5*i+1); records[5*i+4]=consumed;
            }
            output.write(reinterpret_cast<char*>(records.data()),records.size()*8);
        }
        if(!output) throw std::runtime_error("prediction write");
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
