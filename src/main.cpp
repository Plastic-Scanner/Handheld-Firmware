// https://github.com/arduino/ArduinoTensorFlowLiteTutorials/blob/master/GestureToEmoji/ArduinoSketches/IMU_Classifier/IMU_Classifier.ino
// https://colab.research.google.com/drive/1wfOuVHbrcoFD7YLNErialoZrMzzZKGtq#scrollTo=3LlwY8B_h8rJ

#include <Arduino.h>

/////////////////////////db/////////////////////////////
// Include necessary libraries for sensors and modules
#include <Wire.h> // The I2C library
#include <SparkFun_Qwiic_Scale_NAU7802_Arduino_Library.h>
NAU7802 nau; // Create an instance of the NAU7802 sensor
// #include "tlc59208.h"
// TLC59208 ledctrl;
#include "PCA9551.h"
PCA9551 ledDriver(0x60); // Create an instance of the PCA9551 LED driver

/////////////////////////ml/////////////////////////////
// Include TensorFlow Lite and related libraries
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
// #include <tensorflow/lite/version.h>
#include "model (12).h" // Include the machine learning model

// Global variables for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;
TfLiteTensor *tflInputTensor = nullptr;
TfLiteTensor *tflOutputTensor = nullptr;
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Array to map plastic index to a name
const char *PLASTICS[] = {
    "PC",
    "PET",
    "PMMA",
    "other",
    "PS"};
#define NUM_PLASTICS (sizeof(PLASTICS) / sizeof(PLASTICS[0]))

////////////////////////screen//////////////////////
// Include libraries for the OLED display
#include <U8g2lib.h>
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/U8X8_PIN_NONE);

////////////////////////other//////////////////////
// Define threshold values for brightness and darkness
#define TooBright 1.50
#define TooDark 0.5

// sensor readings
float readings[] = {1116508.0, 1540233.0, 14747.0, 14787.0, 14942.0, 15038.0, 14959.0, 14609.0};
float calibrate_readings[] = {1118173.0, 1561292.0, 10868.0, 10924.0, 10904.0, 10878.0, 10923.0, 10916.0};
float normalized[8];
float snv[8];

// Button configuration
const int buttonPin = 26; // the number of the pushbutton pin
int buttonState = 0;      // variable for reading the pushbutton status

// Battery voltage pin
#define VBATPIN A13

// Function to calculate the mean of an array
float calculateMean(float values[], int size)
{
  float sum = 0;
  for (int i = 0; i < size; i++)
  {
    sum += values[i];
  }
  return sum / size;
}

// Function to calculate the standard deviation of an array
float calculateStdDev(float values[], int size)
{
  float mean = calculateMean(values, size);
  float sum = 0;
  for (int i = 0; i < size; i++)
  {
    sum += sq(values[i] - mean);
  }
  return sqrt(sum / (size - 1));
}

// Function to perform sensor scan
void scan()
{
  Serial.println("Starting scan");
  for (int i = 0; i < 8; i++)
  {
    // LED DRIVER: For TLC59208 choose the ledctrl, for PCA9551 choose ledDriver////////////////////
    // ledctrl.on(i);
    ledDriver.digitalWrite(i, LOW); // turns it on
    delay(10);

    // Skip the first 10 readings
    for (uint8_t j = 0; j < 15; j++)
    {
      while (!nau.available())
        delay(1);
      nau.getReading();
    }
    // Read sensor value
    while (!nau.available())
      delay(1);
    readings[i] = nau.getAverage(15);
    // LED DRIVER: For TLC59208 choose the ledctrl, for PCA9551 choose ledDriver////////////////////
    // ledctrl.off(i);
    ledDriver.digitalWrite(i, HIGH); // turns it off
  }

  // Print the sensor readings
  for (int i = 0; i < 8; i++)
  {
    Serial.print(readings[i], 1);
    Serial.print('\t');
  }
  Serial.println();
}

// Function to perform calibration scan
void calibrate_scan()
{
  Serial.println("Start calibration");
  // Loop through each sensor
  for (int i = 0; i < 8; i++)
  {
    // LED DRIVER: For TLC59208 choose the ledctrl, for PCA9551 choose ledDriver////////////////////
    // Serial.println("turning on LED: " + String(i));
    // ledctrl.on(i);
    ledDriver.digitalWrite(i, LOW); // turns it on
    delay(10);

    // Skip the first 10 readings to allow the sensor to stabilize
    for (uint8_t j = 0; j < 15; j++)
    {
      while (!nau.available())
        delay(1);       // Wait for a reading to be available
      nau.getReading(); // Discard the reading
    }

    // Read sensor value (ADC)
    while (!nau.available())
      delay(1);                                 // Wait for a reading to be available
    calibrate_readings[i] = nau.getAverage(15); // Store the calibrated reading
    // LED DRIVER: For TLC59208 choose the ledctrl, for PCA9551 choose ledDriver////////////////////
    // ledctrl.off(i);
    ledDriver.digitalWrite(i, HIGH); // turns it off
  }

  // Print the calibrated readings
  for (int i = 0; i < 8; i++)
  {
    Serial.print(calibrate_readings[i], 1); // Print the reading with 1 decimal place
    Serial.print('\t');                     // Print a tab character to separate readings
  }
  Serial.println(); // Move to the next line in the Serial monitor
}

void setup()
{
  Serial.begin(9600); // Initialize the serial communication
  while (!Serial)
    ; // Wait for Serial to be ready

  /////////////////////////db/////////////////////////////
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(NEOPIXEL_I2C_POWER, OUTPUT);
  digitalWrite(NEOPIXEL_I2C_POWER, HIGH);
  Wire.begin();          // Initialize I2C communication
  Wire.setClock(400000); // Set I2C clock speed to 400kHz

  /////////////////////////ADC/////////////////////////////
  if (!nau.begin())
  {
    Serial.println("Failed to find NAU7802");
    u8g2.setCursor(0, 60);
    u8g2.setFont(u8g2_font_scrum_te);
    u8g2.print("No NAU7802");
    u8g2.sendBuffer(); // Transfer the internal memory to the display
  }

  // Take 10 readings to flush out initial readings
  for (uint8_t i = 0; i < 10; i++)
  {
    while (!nau.available())
      delay(1);       // Wait for a reading to be available
    nau.getReading(); // Discard the reading
  }

  nau.setLDO(NAU7802_LDO_EXTERNAL);   // Set the Low-Dropout Regulator voltage to 3.3V
  nau.setGain(NAU7802_GAIN_1);        // Set the sensor gain to 128
  nau.setSampleRate(NAU7802_SPS_320); // Increase the sample rate to the maximum
  nau.calibrateAFE();                 // Recalibrate the analog front end when settings are changed
  //////////////////////LED/////////////////////////////
  if (ledDriver.begin() == false)
  {
    Serial.println("Could not connect.");
    u8g2.setCursor(0, 50);
    u8g2.setFont(u8g2_font_scrum_te);
    u8g2.print("No LED Driver");
    u8g2.sendBuffer(); // Transfer the internal memory to the display
    while (1)
      ;
  }
  //////////////for later pwm support///////////////////
  // ledDriver.setOutputMode(i, PCA9551_MODE_PWM0); // Set LED0 to PWM mode
  // ledDriver.setPrescaler(i, 43);  //  1 Hz
  // ledDriver.setPWM(i, 128);       //  50% duty cycle

  /////////////////////////ml/////////////////////////////
  // Get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("Model schema mismatch!");
    u8g2.setCursor(0, 40);
    u8g2.setFont(u8g2_font_scrum_te);
    u8g2.print("Model mismatch");
    u8g2.sendBuffer(); // Transfer the internal memory to the display
    while (1)
      ;
  }
  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();
  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  /////////////////////////screen/////////////////////////////
  Serial.println("wait for button press");
  u8g2.setI2CAddress(0x7A);
  u8g2.begin();
  u8g2.clearBuffer();               // Clear the internal memory of the display
  u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
  u8g2.drawStr(0, 16, "Press to");  // Display "Scan" on the screen
  u8g2.drawStr(0, 36, "Calibrate"); // Display "Scan" on the screen
  u8g2.sendBuffer();                // Transfer internal memory to the display
  while (digitalRead(buttonPin) == HIGH)
  {
    // wait for button press
  }
  calibrate_scan();
  // check if the calibration reading is not too far off (that the LEDs do not work)
  u8g2.clearBuffer();               // Clear the internal memory of the display
  u8g2.drawStr(0, 16, "Done!");     // Display "Scan" on the screen
  u8g2.sendBuffer();                // Transfer internal memory to the display
  delay(1000);                      // Wait for 3 seconds before starting calibration
  u8g2.clearBuffer();               // Clear the internal memory of the display
  u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
  u8g2.drawStr(0, 16, "Press to");  // Display "Scan" on the screen
  u8g2.drawStr(0, 36, "Scan");      // Display "Scan" on the screen
  int measuredvbat = analogReadMilliVolts(VBATPIN);
  int batteryVoltage = map(measuredvbat, 1500, 2100, 0, 100); // Map the voltage to a percentage (0-100%)
  u8g2.setCursor(70, 60);
  u8g2.setFont(u8g2_font_scrum_te);
  u8g2.print("Bat:");
  u8g2.print(batteryVoltage); // Display likelihood as a percentage
  u8g2.print("%");
  u8g2.sendBuffer(); // Transfer the internal memory to the display
}

void loop()
{
  // Wait for input to start the scan
  // check if the pushbutton is pressed. If it is, the buttonState is HIGH:

  if (digitalRead(buttonPin) == LOW)
  {
    /////////////////////////screen/////////////////////////////
    Serial.println("Start SCAN");
    scan();
    /////// preprocess data
    // Normalize
    for (int i = 0; i < 8; i++)
    {
      normalized[i] = (float)readings[i] / (float)calibrate_readings[i];
    }

    // Check quality of the sample
    if (normalized[0] > TooBright)
    {
      Serial.println("Sample too bright");
      u8g2.clearBuffer();               // Clear the internal memory of the display
      u8g2.drawStr(0, 16, "Incorrect"); // Display "Too Bright" on the screen
      u8g2.sendBuffer();                // Transfer internal memory to the display
      delay(2000);                      // Wait for 2 seconds
    }
    else if (normalized[0] < TooDark)
    {
      Serial.println("Sample too dark");
      u8g2.clearBuffer();              // Clear the internal memory of the display
      u8g2.drawStr(0, 16, "Too Dark"); // Display "Too Dark" on the screen
      u8g2.sendBuffer();               // Transfer internal memory to the display
      delay(2000);                     // Wait for 2 seconds
    }
    else
    {
      // SNV transform
      float mean = calculateMean(normalized, 8);  // Calculate the mean of normalized values
      float std = calculateStdDev(normalized, 8); // Calculate the standard deviation of normalized values

      // Apply SNV (Standard Normal Variate) transformation
      for (int i = 0; i < 8; i++)
      {
        snv[i] = (normalized[i] - mean) / std;
        // snv[i] = snv[i] * 1000000; // Scale the SNV values to 0-1000 to get more resolution
      }

      ////// Run TensorFlow inference
      for (int i = 0; i < 8; i++)
      {
        tflInputTensor->data.f[i] = snv[i]; // Set input tensor values with SNV-transformed data
      }
      TfLiteStatus invokeStatus = tflInterpreter->Invoke(); // Run the inference
      if (invokeStatus != kTfLiteOk)
      {
        Serial.println("Invoke failed!"); // Print an error message if inference fails
        while (1)
          ; // Infinite loop to halt execution
        return;
      }

      ////// Output the results
      float maxLikelihood = -1.0; // Initialize with a negative value
      int maxLikelihoodIndex = -1;
      u8g2.clearBuffer(); // Clear the internal memory of the display
      u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
      // Loop through plastic types
      for (int i = 0; i < NUM_PLASTICS; i++)
      {
        Serial.print(PLASTICS[i]);
        Serial.print(": ");
        Serial.println(tflOutputTensor->data.f[i], 6); // Print the likelihood with 6 decimal places

        // Check if the current plastic has a higher likelihood
        if (tflOutputTensor->data.f[i] > maxLikelihood)
        {
          maxLikelihood = tflOutputTensor->data.f[i];
          maxLikelihoodIndex = i;
        }
      }

      if (maxLikelihoodIndex != -1)
      {
        // Print the most likely plastic type
        Serial.print("Most likely plastic: ");
        Serial.println(PLASTICS[maxLikelihoodIndex]);

        // Display the most likely plastic type and its likelihood on the screen
        u8g2.drawStr(0, 32, PLASTICS[maxLikelihoodIndex]);
        u8g2.setCursor(72, 32);
        u8g2.print(int(tflOutputTensor->data.f[maxLikelihoodIndex] * 100)); // Display likelihood as a percentage
        u8g2.print("%");
      }
      u8g2.sendBuffer(); // Transfer the internal memory to the display

      Serial.println("done");
      Serial.println();
      delay(4000);                      // Wait for 5 seconds before the next iteration
      u8g2.clearBuffer();               // Clear the internal memory of the display
      u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
      u8g2.drawStr(0, 16, "Press to");  // Display "Scan" on the screen
      u8g2.drawStr(0, 36, "Scan");      // Display "Scan" on the screen
      int measuredvbat = analogReadMilliVolts(VBATPIN);
      int batteryVoltage = map(measuredvbat, 1500, 2100, 0, 100); // Map the voltage to a percentage (0-100%)
      u8g2.setCursor(70, 60);
      u8g2.setFont(u8g2_font_scrum_te);
      u8g2.print("Bat:");
      u8g2.print(batteryVoltage); // Display likelihood as a percentage
      u8g2.print("%");
      u8g2.sendBuffer(); // Transfer the internal memory to the display
    }
  }
  else
  {
  }
}
