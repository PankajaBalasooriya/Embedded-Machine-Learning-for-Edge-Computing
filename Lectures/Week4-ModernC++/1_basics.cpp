#include <iostream>
#include <string>

// Function dclaration
void get_show_hobbies();

// Entry point for control(where program starts to execute)
// int return type, return 0 when successfully executed
int main(){
    int age;
    char greeting[] = "Welcome!\n"; // initialization
    std::string name; // type declaration
    // :: scope resolution operator

    std::cout << greeting; // Console Out
    std::cout << "Name: ";
    std::cin >>name;    // console in
    std::cout << "Age: ";
    std::cin >> age;
    std::cout << "You are: ";
    std::cout << name << ", " << age;

    if (age < 5)
        std::cout << ", What are you foing here?\n\n";
    else if (age < 18)
        std::cout << ", You cannot vote!\n\n";
    else
        std::cout << ", You can vote!\n\n";
    
    // Function call
    get_show_hobbies();
    return 0;
}
// Function definition
void get_show_hobbies(){
    // std - C++ standard library
    using namespace std; // within the scope of the function using all the functions within std
    // This should be done only in small functions
    // It is not a good practive to use namespace in top of the main to avoid name collisions
    string hobbies[3];
    for(int i=0; i<3; i++){
        cout<< "Hobby " << i+1 << ": ";
        cin >> hobbies[i];
    }
    // range based for loops - for each loop
    for(auto & hobby: hobbies) // auto - automatic type declaratrons // & - reference - read the values insted of copying the whole thing passing the address. not complicated as a pointer
        cout << hobby << ", ";
}

// avoid uninitialized vairables - move it closer to the place where youa are using it 
// std::string is preferd over char[]
// Auto can make code more readable or less readable depending on the usage
// restrict using namespaces to limites scopes
