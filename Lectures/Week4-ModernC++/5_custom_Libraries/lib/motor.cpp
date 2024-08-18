#include "Arduino.h"
#include "motor.h"

Motor::Motor(String name, int pin_1, int pin_2, int pin_pwm){
    this->name = name;
    this->pin_1 = pin_1;
    this->pin_2 = pin_2;
    this->pin_pwm = pin_pwm;

    pinMode(pin_1, OUTPUT);
    pinMode(pin_2, OUTPUT);
}

void Motor::forward(int speed){
    this->speed = speed;
    this->is_forward = true;

    digitalWrite(pin_1, HIGH);
    digitalWrite(pin_2, LOW);
    analogWrite(pin_pwm, speed);

    Serial.println(name + " : " + speed + " : " + "Forward");
}

void Motor::reverse(int speed){
    this->speed = speed;
    this->is_forward = false;

    digitalWrite(pin_1, LOW);
    digitalWrite(pin_2, HIGH);
    analogWrite(pin_pwm, speed);

    Serial.println(name + " : " + speed + " : " + "Reverse");
}