#include <Arduino.h>

// Classes have zero overhead
class Motor{
    private: // Private attributes: can only be accessed by the methods of the class
        int pin_1, pin_2, pin_pwm;
    
    public: // Public attributes: can be accessed by any part of the code
        String name;
        int speed;
        bool is_forward;

    Motor(String name, int pin_1, int pin_2, int pin_pwm):  // constructors -  initialize the object
        name(name), pin_1(pin_1), pin_2(pin_2), pin_pwm(pin_pwm){ // initialize the attributes
            pinMode(pin_1, OUTPUT);
            pinMode(pin_2, OUTPUT);
        } 
    // member functions
    void forward(int speed){
      this->speed = speed; // this-> is a pointer to the object itself. to avoid confusion between the attribute and the parameter
      this->is_forward = true;

      digitalWrite(pin_1, HIGH);
      digitalWrite(pin_2, LOW);
      analogWrite(pin_pwm, speed);

      Serial.println(name + " : " + speed + " : " + "Forward");
    }

    void reverse(int);  // Member function declared inside the definition but defined outside out side.
};

void Motor::reverse(int speed){
  this->speed = speed;
  this->is_forward = false;

  digitalWrite(pin_1, LOW);
  digitalWrite(pin_2, HIGH);
  analogWrite(pin_pwm, speed);

  Serial.println(name + " : " + speed + " : " + "Reverse");
}


// ----------------------------
Motor motor_right("right", 5,6, 7);
Motor motor_left("left", 8,9, 10);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

}

void loop() {
  // put your main code here, to run repeatedly:
  motor_right.forward(55);
  delay(2000);
  motor_right.reverse(100);
  delay(2000);
  motor_right.forward(255);
  delay(2000);
  motor_right.reverse(0);
  delay(2000);
}

