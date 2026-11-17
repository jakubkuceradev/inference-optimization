#pragma once
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

// Paper-derived recipes and project extension are attributed in feature_comparison.json.
class Features {
    std::string recipe;
    size_t length, position=0;
    std::vector<float> ring;
    std::array<size_t,4> windows{5,20,100,384};
    std::array<std::array<double,144>,4> sums{};
    const float* lag(size_t n) const { return ring.data()+((position+length-1-n)%length)*144; }
    void mean5(float* output) const {
        for(size_t f=0;f<144;++f) {
            double value=0.;
            for(int n=4;n>=0;--n) value+=lag(n)[f];
            output[f]=static_cast<float>(value/5.);
        }
    }
public:
    explicit Features(const std::string& name):recipe(name),length(name=="current"?1:name=="multiscale384"?384:5),ring(length*144) {
        if(name!="current" && name!="mean5" && name!="current_mean5" && name!="concat5" && name!="multiscale384")
            throw std::runtime_error("Unknown feature recipe");
    }
    size_t history() const { return length; }
    size_t dimension() const {
        if(recipe=="current" || recipe=="mean5") return 144;
        if(recipe=="current_mean5") return 288;
        return recipe=="concat5"?720:1440;
    }
    void reset(const float* chronological_history) {
        std::copy_n(chronological_history,length*144,ring.begin()); position=0;
        if(recipe=="multiscale384") {
            for(size_t w=0;w<4;++w) {
                sums[w].fill(0.);
                for(size_t j=length-windows[w];j<length;++j)
                    for(size_t f=0;f<144;++f) sums[w][f]+=ring[j*144+f];
            }
        }
    }
    void emit(float* output) const {
        if(recipe=="current") { std::copy_n(lag(0),144,output); return; }
        if(recipe=="mean5") { mean5(output); return; }
        std::copy_n(lag(0),144,output);
        if(recipe=="current_mean5") { mean5(output+144); return; }
        if(recipe=="concat5") {
            for(size_t n=1;n<5;++n) std::copy_n(lag(n),144,output+n*144);
            return;
        }
        for(size_t w=0;w<4;++w)
            for(size_t f=0;f<144;++f) output[(w+1)*144+f]=static_cast<float>(sums[w][f]/windows[w]);
        const std::array<size_t,5> lags{1,4,19,99,383};
        for(size_t n=0;n<5;++n) std::copy_n(lag(lags[n]),144,output+(n+5)*144);
    }
    void update(const float* incoming,float* output) {
        if(recipe=="current") { std::copy_n(incoming,144,output); return; }
        if(recipe=="multiscale384") {
            for(size_t w=0;w<4;++w) {
                const float* departing=ring.data()+((position+length-windows[w])%length)*144;
                for(size_t f=0;f<144;++f) { sums[w][f]-=departing[f]; sums[w][f]+=incoming[f]; }
            }
        }
        std::copy_n(incoming,144,ring.data()+position*144);
        position=(position+1)%length; emit(output);
    }
};
