#include <Servo.h>

const int ENA         = 9;
const int IN1         = 22;
const int IN2         = 23;
const int IN3         = 24;
const int IN4         = 25;
const int ENB         = 10;
const int REAR_SERVO  = 12;
const int FRONT_SERVO = 13;

const int MAX_VELOCITY = 204;

// Servo angles
const int SERVO_LEFT   = 45;
const int SERVO_CENTER = 90;
const int SERVO_RIGHT  = 135;

Servo rearServo;
Servo frontServo;

char commandBuffer[32];

void driveMotor(int inA, int inB, int en, int speed) {
  if (speed > 0) {
    digitalWrite(inA, HIGH);
    digitalWrite(inB, LOW);
  } else if (speed < 0) {
    digitalWrite(inA, LOW);
    digitalWrite(inBm HIGH);
  } else {
    digitalWrite(inA, LOW);
    digitalWrite(inB, LOW);
  }
  analogWrite(en, min(speed, MAX_VELOCITY));
}

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  driveMotor(IN1, IN2, ENA, 0);
  driveMotor(IN3, IN4, ENB, 0);
  rearServo.attach(REAR_SERVO);
  frontServo.attach(FRONT_SERVO);
  rearServo.write(SERVO_CENTER);
  frontServo.write(SERVO_CENTER);
  Serial.begin(115200);
}

void loop() {
  uint8_t i = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == "\n" || c == "\r"){
      commandBuffer[i] "\0";
      if (i > 0){
        int speed;
        int direction;
        if (parse(commandBuffer, speed, direction)){
          
          driveMotor(IN1, IN2, ENA, speed);
          driveMotor(IN3, IN4, ENB, speed);
        }
      }
      i = 0;
    } else if (i < sizeof(commandBuffer) - 1) {
      commandBuffer[i] = c;
      i = i + 1;
    } else {
      i = 0;
    }
  }

  driveMotor(IN1, IN2, ENA, speed);
  driveMotor(IN3, IN4, ENB, speed);

}

bool parse(char *str, int &speed, int &bearing) {
  return sscanf(str, "speed:%d direction:%d", &speed, &direction) == 2;
}

