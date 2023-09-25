//https://github.com/arduino/ArduinoTensorFlowLiteTutorials/blob/master/GestureToEmoji/ArduinoSketches/IMU_Classifier/IMU_Classifier.ino
//https://colab.research.google.com/drive/1wfOuVHbrcoFD7YLNErialoZrMzzZKGtq#scrollTo=3LlwY8B_h8rJ
#include <Arduino.h>

/////////////////////////db/////////////////////////////
// Include necessary libraries for sensors and modules
#include <Wire.h>     // The I2C library
#include <SparkFun_Qwiic_Scale_NAU7802_Arduino_Library.h>
NAU7802 nau; // Create an instance of the NAU7802 sensor
#include "PCA9551.h"
PCA9551 ledDriver = PCA9551(PCA9551_ADDR_1); // Create an instance of the PCA9551 LED driver

/////////////////////////ml/////////////////////////////
// Include TensorFlow Lite and related libraries
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
// #include <tensorflow/lite/version.h>
#include "model.h" // Include the machine learning model

// Global variables for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Array to map gesture index to a name
const char* PLASTICS[] = {
  "other",
  "PMMA",
  "PS",
  "PET",
  "PC"
};
#define NUM_PLASTICS (sizeof(PLASTICS) / sizeof(PLASTICS[0]))

////////////////////////screen//////////////////////
// Include libraries for the OLED display
#include <U8g2lib.h>
#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

////////////////////////other//////////////////////
// Define threshold values for brightness and darkness
#define TooBright 1.4
#define TooDark 0.6

float readings[] = {22977, 36106, 52788, 71216, 27235, 35036, 10490, 6381};
float calibrate_readings[] = {26016, 46824, 82300, 80176, 42096, 53390, 19076, 13274};
float normalized[8];
float snv[8];

const int buttonPin = 8;  // the number of the pushbutton pin
int buttonState = 0;  // variable for reading the pushbutton status


// Function to calculate the mean of an array
float calculateMean(float values[], int size) {
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += values[i];
  }
  return sum / size;
}

// Function to calculate the standard deviation of an array
float calculateStdDev(float values[], int size) {
  float mean = calculateMean(values, size);
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += sq(values[i] - mean);
  }
  return sqrt(sum / (size - 1));
}

// Function to perform sensor scan
void scan(){
  Serial.println("Starting scan");
  for (int i=0; i<8; i++) {
      ledDriver.setLedState(i, LED_ON);
      delay(10);

      // Skip the first 10 readings
      for (uint8_t j=0; j<10; j++) {
          while (! nau.available()) delay(1);
          nau.getReading();
      }
      // Read sensor value
      while (! nau.available()) delay(1);
      readings[i] = nau.getAverage(10);
      ledDriver.setLedState(i, LED_OFF);
  }

  // Print the sensor readings
  for (int i=0; i<8; i++) {
      Serial.print(readings[i], 1);
      Serial.print('\t');
  }
  Serial.println();
}

// Function to perform calibration scan
void calibrate_scan() {
  Serial.println("Start calibration");
  // Loop through each sensor
  for (int i = 0; i < 8; i++) {
    ledDriver.setLedState(i, LED_ON); // Turn on the LED for this sensor
    delay(10);

    // Skip the first 10 readings to allow the sensor to stabilize
    for (uint8_t j = 0; j < 10; j++) {
      while (!nau.available()) delay(1); // Wait for a reading to be available
      nau.getReading(); // Discard the reading
    }

    // Read sensor value (ADC)
    while (!nau.available()) delay(1); // Wait for a reading to be available
    calibrate_readings[i] = nau.getAverage(10); // Store the calibrated reading
    ledDriver.setLedState(i, LED_OFF); // Turn off the LED for this sensor
  }

  // Print the calibrated readings
  for (int i = 0; i < 8; i++) {
    Serial.print(calibrate_readings[i], 1); // Print the reading with 1 decimal place
    Serial.print('\t'); // Print a tab character to separate readings
  }
  Serial.println(); // Move to the next line in the Serial monitor
}

void setup() {
  Serial.begin(9600); // Initialize the serial communication
  while (!Serial); // Wait for Serial to be ready
  pinMode(buttonPin, INPUT_PULLUP);

  /////////////////////////db/////////////////////////////
  Wire.begin(); // Initialize I2C communication
  Wire.setClock(400000); // Set I2C clock speed to 400kHz

  if (!nau.begin()) {
    Serial.println("Failed to find NAU7802");
  }

  // Take 10 readings to flush out initial readings
  for (uint8_t i = 0; i < 10; i++) {
    while (!nau.available()) delay(1); // Wait for a reading to be available
    nau.getReading(); // Discard the reading
  }

  nau.setLDO(3.3); // Set the Low-Dropout Regulator voltage to 3.3V
  nau.setGain(NAU7802_GAIN_128); // Set the sensor gain to 128
  nau.setSampleRate(NAU7802_SPS_320); // Increase the sample rate to the maximum
  nau.calibrateAFE(); // Recalibrate the analog front end when settings are changed

  /////////////////////////ml/////////////////////////////
  // Get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);

  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  /////////////////////////screen/////////////////////////////

  Serial.println("Starting calibration in 3 seconds");
  u8g2.begin();
  u8g2.clearBuffer(); // Clear the internal memory of the display
  u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
  u8g2.drawStr(0, 16, "Press to"); // Display "Scan" on the screen
  u8g2.drawStr(0, 36, "Calibrate"); // Display "Scan" on the screen
  u8g2.sendBuffer(); // Transfer internal memory to the display
  while (digitalRead(buttonPin) == HIGH) {
    // wait for button press
  }
  calibrate_scan();
  u8g2.clearBuffer(); // Clear the internal memory of the display
  u8g2.drawStr(0, 16, "Done!"); // Display "Scan" on the screen
  u8g2.sendBuffer(); // Transfer internal memory to the display
  delay(1000); // Wait for 3 seconds before starting calibration
}

void loop() {
  // Wait for input to start the scan
  // check if the pushbutton is pressed. If it is, the buttonState is HIGH:
  u8g2.clearBuffer(); // Clear the internal memory of the display
  u8g2.drawStr(0, 16, "Press to"); // Display "Scan" on the screen
  u8g2.drawStr(0, 36, "Scan"); // Display "Scan" on the screen
  u8g2.sendBuffer(); // Transfer internal memory to the display
  if (digitalRead(buttonPin) == LOW) {
    /////////////////////////screen/////////////////////////////
    Serial.println("Start SCAN in 3 seconds");
    scan();
    /////// preprocess data
    // Normalize
    for (int i = 0; i < 8; i++) {
      normalized[i] = (float)readings[i] / (float)calibrate_readings[i];
    }

    // Check quality of the sample
    if (normalized[0] > TooBright) {
      Serial.println("Sample too bright");
      u8g2.clearBuffer(); // Clear the internal memory of the display
      u8g2.drawStr(0, 16, "Too Bright"); // Display "Too Bright" on the screen
      u8g2.sendBuffer(); // Transfer internal memory to the display
      delay(2000); // Wait for 2 seconds
    } else if (normalized[0] < TooDark) {
      Serial.println("Sample too dark");
      u8g2.clearBuffer(); // Clear the internal memory of the display
      u8g2.drawStr(0, 16, "Too Dark"); // Display "Too Dark" on the screen
      u8g2.sendBuffer(); // Transfer internal memory to the display
      delay(2000); // Wait for 2 seconds
    }
    else {
      // SNV transform
      float mean = calculateMean(normalized, 8); // Calculate the mean of normalized values
      float std = calculateStdDev(normalized, 8); // Calculate the standard deviation of normalized values

      // Apply SNV (Standard Normal Variate) transformation
      for (int i = 0; i < 8; i++) {
        snv[i] = (normalized[i] - mean) / std;
      }

      ////// Run TensorFlow inference
      for (int i = 0; i < 8; i++) {
        tflInputTensor->data.f[i] = snv[i]; // Set input tensor values with SNV-transformed data
      }
      TfLiteStatus invokeStatus = tflInterpreter->Invoke(); // Run the inference
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Invoke failed!"); // Print an error message if inference fails
        while (1); // Infinite loop to halt execution
        return;
      }

      ////// Output the results
      float maxLikelihood = -1.0; // Initialize with a negative value
      int maxLikelihoodIndex = -1;
      u8g2.clearBuffer(); // Clear the internal memory of the display

      // Loop through plastic types
      for (int i = 0; i < NUM_PLASTICS; i++) {
        Serial.print(PLASTICS[i]);
        Serial.print(": ");
        Serial.println(tflOutputTensor->data.f[i], 6); // Print the likelihood with 6 decimal places

        // Check if the current plastic has a higher likelihood
        if (tflOutputTensor->data.f[i] > maxLikelihood) {
          maxLikelihood = tflOutputTensor->data.f[i];
          maxLikelihoodIndex = i;
        }
      }

      if (maxLikelihoodIndex != -1) {
        // Print the most likely plastic type
        Serial.print("Most likely plastic: ");
        Serial.println(PLASTICS[maxLikelihoodIndex]);

        // Display the most likely plastic type and its likelihood on the screen
        u8g2.drawStr(0, 32, PLASTICS[maxLikelihoodIndex]);
        u8g2.setCursor(64, 32);
        u8g2.print(int(tflOutputTensor->data.f[maxLikelihoodIndex] * 100)); // Display likelihood as a percentage
        u8g2.print("%");
      }
      u8g2.sendBuffer(); // Transfer the internal memory to the display
      Serial.println("done");
      Serial.println();
      delay(5000); // Wait for 5 seconds before the next iteration
    }
  }  
  else {
  }
}
