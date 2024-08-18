#include <Arduino.h>
#include "motor.h"

Motor motor_left("left", 2, 3, 4);
Motor motor_right("right", 5, 6, 7);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:
  motor_left.forward(255);
  motor_right.forward(255);
  delay(2000);
  
  motor_left.reverse(255);
  motor_right.reverse(255);
  delay(2000);
  
}

