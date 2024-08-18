#include <iostream>

int main(){
    float pi_f = 3.14159;
    std::cout << "pi float: " << pi_f << std::endl;
    // pi float: 3.14159

    int radius_i = 10;
    std::cout << "2*pi*r: " << 2* pi_f * radius_i << std::endl; // implicit casting : int-> float
    // 2*pi*r: 62.8318

    int pi_si = pi_f; // implicit casting
    pi_si = (int)pi_f; // explicti casting
    std::cout << "pi int: " << pi_si << std::endl;
    // pi int: 3
    // not rounding off, 3 because of truncation

    int *ptr_i = (int*)0x1111;
    std::cout << "ptr_i: " << ptr_i << std::endl;
    // ptr_i: 0x1111
}