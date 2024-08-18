// Header guard (outdated)
// #ifndef Motor_h
// #define Motor_h 

#pragma once // Header guard - to prevent # including same header multipe times

# include <Arduino.h>

// only declarations
// No definitions
class Motor{
    private: 
        int pin_1, pin_2, pin_pwm;
        int speed;
        bool is_forward;
    
    public: 
        String name;

    Motor (String name, 
            int pin_1, 
            int pin_2, 
            int pin_pwm);
    void forward(int speed);
    void reverse(int speed);
};

// #endif