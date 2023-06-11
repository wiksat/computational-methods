#include <cmath>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
using namespace std;

float* prepare_floats(){
    static float floats[101];
    float left=0.99;
    float right=1.01;
    float span=(right-left)/100;
    for (int i = 0; i < 101; ++i) {
        floats[i]=left+i*span;
    }
    return floats;
}

double* prepare_doubles(){
    static double doubles[101];
    double left2=0.99;
    double right2=1.01;
    double span2=(right2-left2)/100;
    for (int i = 0; i < 101; ++i) {
        doubles[i]=left2+i*span2;
    }
    return doubles;
}

long double* prepare_long_doubles(){
    static long double long_doubles[101];
    long double left3=0.99;
    long double right3=1.01;
    long double span3=(right3-left3)/100;
    for (int i = 0; i < 101; ++i) {
        long_doubles[i]=left3+i*span3;
    }
    return long_doubles;
}

int main() {
    float* floats;
    double* doubles;
    long double* long_doubles;

    floats = prepare_floats();
    doubles = prepare_doubles();
    long_doubles = prepare_long_doubles();

    std::cout<< "Wzor 1"<< std::endl;
    std::cout<< "float - doucle - long double"<< std::endl;
    for (int i = 0; i < 101; ++i) {
        float f;
        double d;
        long double ld;
        f=powf(floats[i],8)-8*powf(floats[i],7)+28*powf(floats[i],6)-56*powf(floats[i],5)+70*powf(floats[i],4)-56*powf(floats[i],3)+28*powf(floats[i],2)-8*floats[i]+1;
        d=pow(doubles[i],8)-8*pow(doubles[i],7)+28*pow(doubles[i],6)-56*pow(doubles[i],5)+70*pow(doubles[i],4)-56*pow(doubles[i],3)+28*pow(doubles[i],2)-8*doubles[i]+1;
        ld=powl(long_doubles[i],8)-8*powl(long_doubles[i],7)+28*powl(long_doubles[i],6)-56*powl(long_doubles[i],5)+70*powl(long_doubles[i],4)-56*powl(long_doubles[i],3)+28*powl(long_doubles[i],2)-8*long_doubles[i]+1;
        std::cout<<std::fixed<< std::setprecision(30) <<"  "<< f << "    "<< d << "    "<< ld << "    "<< std::endl;
//        std::cout<< std::setprecision(30) << f << "    "<< d << "    "<< ld << "    "<< std::endl;
    }
    std::cout<< "Wzor 2"<< std::endl;
    std::cout<< "float - doucle - long double"<< std::endl;
    for (int i = 0; i < 101; ++i) {
        float f;
        double d;
        long double ld;
        f=(((((((floats[i]-8)*floats[i]+28)*floats[i]-56)*floats[i]+70)*floats[i]-56)*floats[i]+28)*floats[i]-8)*floats[i]+1;
        d=(((((((doubles[i]-8)*doubles[i]+28)*doubles[i]-56)*doubles[i]+70)*doubles[i]-56)*doubles[i]+28)*doubles[i]-8)*doubles[i]+1;
        ld=(((((((long_doubles[i]-8)*long_doubles[i]+28)*long_doubles[i]-56)*long_doubles[i]+70)*long_doubles[i]-56)*long_doubles[i]+28)*long_doubles[i]-8)*long_doubles[i]+1;
        std::cout<<std::fixed<< std::setprecision(30)<<  f << "    "<< d << "    "<< ld << "    "<< std::endl;
//        std::cout<< std::setprecision(30)<<  f << "    "<< d << "    "<< ld << "    "<< std::endl;
    }
    std::cout<< "Wzor 3"<< std::endl;
    std::cout<< "float - doucle - long double"<< std::endl;
    for (int i = 0; i < 101; ++i) {
        float f;
        double d;
        long double ld;
        f=powf((floats[i]-1),8);
        d=pow((doubles[i]-1),8);
        ld=powl((long_doubles[i]-1),8);
        std::cout<<std::fixed<<std::setprecision(30) << f << "    "<< d << "    "<< ld << "    "<< std::endl;
//        std::cout<<std::setprecision(30) << f << "    "<< d << "    "<< ld << "    "<< std::endl;
    }
    std::cout<< "Wzor 4"<< std::endl;
    std::cout<< "float - doucle - long double"<< std::endl;
    for (int i = 0; i < 101; ++i) {
        float f;
        double d;
        long double ld;
        float absToF=std::abs(floats[i]-1);
        float logToF=std::log(absToF);
        f= std::exp(8* logToF);

        double absToD=std::abs(doubles[i]-1);
        d= exp(8* log(absToD));

        long double absToLD=std::abs(long_doubles[i]-1);
        long double logToLD=std::log(absToLD);
        ld=std::exp(8* logToLD);
        std::cout<<std::fixed <<std::setprecision(30) <<  f << "    "<< d << "    "<< ld << "    "<< std::endl;
//        std::cout<<std::setprecision(30) <<  f << "    "<< d << "    "<< ld << "    "<< std::endl;
    }

    return 0;
}
