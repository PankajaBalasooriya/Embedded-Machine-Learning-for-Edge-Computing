#include <iostream>
int main(){
    int var_si = 1;
    unsigned int var_ui = 5;

    std::cout << "int bytes: " << sizeof(int) << std::endl;
    // int bytes = 4

    var_si = -2147483648;
    std::cout << "si_min: " << var_si << std::endl;
    // var_si = -214748364

    var_si = var_si - 1;
    std::cout << "si_min - 1: " << var_si << std::endl;
    // si_min - 1: 2147483647

    unsigned long long var_ull = 100000000000;
    std::cout << "ull: " << var_ull << std::endl;
    // ull: 100000000000

    var_si = var_ull;
    std::cout << "implicit ull->si: " << var_si << std::endl; // Accidental implicit casting to a smaller datatype
    // implicit ull->si: 1215752192  //  last 4 bits removed 



}