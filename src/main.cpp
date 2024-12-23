// https://github.com/arduino/ArduinoTensorFlowLiteTutorials/blob/master/GestureToEmoji/ArduinoSketches/IMU_Classifier/IMU_Classifier.ino
// https://colab.research.google.com/drive/1wfOuVHbrcoFD7YLNErialoZrMzzZKGtq#scrollTo=3LlwY8B_h8rJ

#include <Arduino.h>
#include <WiFi.h>
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
#include "2024-11-03_model.h" // Include the machine learning model
#include "secrets.h"          // Include the machine learning model
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
    "PLA",
    "PETG",
    "TPU",
    "other"};
#define NUM_PLASTICS (sizeof(PLASTICS) / sizeof(PLASTICS[0]))

////////////////////////screen//////////////////////
// Include libraries for the OLED display
#include <U8g2lib.h>
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/U8X8_PIN_NONE);

////////////////////////other//////////////////////
// Define threshold values for brightness and darkness
#define TooBright 1.90
#define TooDark 0.1
#define offset 0.02
float scaler = 1000000;
// sensor readings
float readings[8];
float calibrate_readings[8];
float normalized[8];
float snv[8];
float background[2];
float snvScaled[8];
float minSNV = -2.5 * scaler;
float maxSNV = 2.5 * scaler;
// Button configuration
const int buttonPin = 26; // the number of the pushbutton pin
int buttonState = 0;      // variable for reading the pushbutton status
bool isScanning = false;  // Variable to keep track of scanning state
int lastLikelihoodIndex = -1;
int consecutiveCount = 0;
bool update = false;
bool sample = false;
// Battery voltage pin
#define VBATPIN A13
const unsigned long shortPressTime = 1000; // 2000 milliseconds = 2 seconds
bool scanMode = false;

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

// Function to skip the first "Count" readings and read the same "Count" average of the sensor value
float readSensorValue(uint8_t Count)
{
  // Skip the first skipCount readings
  for (uint8_t j = 0; j < Count; j++)
  {
    while (!nau.available())
      delay(1);
    nau.getReading();
  }
  // Read sensor value
  while (!nau.available())
    delay(1);
  return nau.getAverage(Count);
}

// Function to perform sensor scan
void performScan(bool isCalibration)
{
  Serial.println("Starting scan");
  // Pre Scan
  background[0] = readSensorValue(10);

  // Actual Scan
  for (int i = 0; i < 8; i++)
  {
    // LED DRIVER: For TLC59208 choose the ledctrl, for PCA9551 choose ledDriver////////////////////
    // ledctrl.on(i);
    ledDriver.digitalWrite(i, LOW); // turns it on
    delay(10);

    float sensorValue = readSensorValue(10);

    if (isCalibration)
    {
      calibrate_readings[i] = sensorValue;
    }
    else
    {
      readings[i] = sensorValue;
    }

    ledDriver.digitalWrite(i, HIGH); // turns it off
  }

  // Post Scan
  background[1] = readSensorValue(10);

  float backgroundAverage = (background[0] + background[1]) / 2;

  for (int i = 0; i < 8; i++)
  {
    if (isCalibration)
    {
      calibrate_readings[i] -= backgroundAverage;
      Serial.print(calibrate_readings[i], 1); // Print the sensor readings
    }
    else
    {
      readings[i] -= backgroundAverage;
      Serial.print(readings[i], 1); // Print the sensor readings
    }
    Serial.print('\t');
  }
  Serial.println();
}

bool preprocess()
{
  // Normalize
  for (int i = 0; i < 8; i++)
  {
    normalized[i] = (float)readings[i] / (float)calibrate_readings[i];
    normalized[i] = normalized[i] * scaler; // Scale the normalized values to 0-1000 to get more resolution
  }
  u8g2.clearBuffer(); // Clear the internal memory of the display

  // Check quality of the sample
  if (normalized[0] > (TooBright * scaler))
  {
    Serial.println("Sample too bright");
    u8g2.drawStr(0, 16, "Incorrect"); // Display "Too Bright" on the screen
    return false;
  }
  else if (normalized[0] < (TooDark * scaler))
  {
    Serial.println("Sample too dark");
    u8g2.drawStr(0, 16, "Too Dark"); // Display "Too Dark" on the screen
    return false;
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
      snv[i] = snv[i] * scaler; // Scale the SNV values to 0-1000 to get more resolution
    }
    ///////////upload new scan//////////
    // scale from +2.5 and -2.5
    for (int i = 0; i < 8; i++)
    {
      snvScaled[i] = (snv[i] - minSNV) / (maxSNV - minSNV);
      snvScaled[i] = snvScaled[i] * scaler; // Scale the SNV values to 0-1000 to get more resolution
    }
    return true;
  }
}

void initWifi()
{
  Serial.print("Connecting to: ");
  Serial.print(ssid);
  WiFi.begin(ssid, password);

  int timeout = 10 * 4; // 10 seconds
  while (WiFi.status() != WL_CONNECTED && (timeout-- > 0))
  {
    delay(250);
    Serial.print(".");
  }
  Serial.println("");

  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("Failed to connect, going back to sleep");
  }

  Serial.print("WiFi connected in: ");
  Serial.print(millis());
  Serial.print(", IP address: ");
  Serial.println(WiFi.localIP());
}

// Make an HTTP request to the IFTTT web service
void makeIFTTTRequest()
{
  String object_test = String("value1:") + readings[0] + ";" + readings[1] + ";" + readings[2] + ";" + readings[3] + ";" + readings[4] + ";" + readings[5] + ";" + readings[6] + ";" + readings[7] + ";" + normalized[0] + ";" + normalized[1] + ";" + normalized[2] + ";" + normalized[3] + ";" + normalized[4] + ";" + normalized[5] + ";" + normalized[6] + ";" + normalized[7] + ";" + snv[0] + ";" + snv[1] + ";" + snv[2] + ";" + snv[3] + ";" + snv[4] + ";" + snv[5] + ";" + snv[6] + ";" + snv[7] + ";" + snvScaled[0] + ";" + snvScaled[1] + ";" + snvScaled[2] + ";" + snvScaled[3] + ";" + snvScaled[4] + ";" + snvScaled[5] + ";" + snvScaled[6] + ";" + snvScaled[7];
  Serial.print(object_test);
  // Serial.print("Connecting to ");
  // Serial.print(server);

  // WiFiClient client;
  // int retries = 5;
  // while (!!!client.connect(server, 80) && (retries-- > 0))
  // {
  //   Serial.print(".");
  // }
  // Serial.println();
  // if (!!!client.connected())
  // {
  //   Serial.println("Failed to connect...");
  // }

  // Serial.print("Request resource: ");
  // Serial.println(resource);

  // // Temperature in Celsius
  // String jsonObject = String("{\"value1\":\"") + readings[0] + ";" + readings[1] + ";" + readings[2] + ";" + readings[3] + ";" + readings[4] + ";" + readings[5] + ";" + readings[6] + ";" + readings[7] + ";" + normalized[0] + ";" + normalized[1] + ";" + normalized[2] + ";" + normalized[3] + ";" + normalized[4] + ";" + normalized[5] + ";" + normalized[6] + ";" + normalized[7] + ";" + snv[0] + ";" + snv[1] + ";" + snv[2] + ";" + snv[3] + ";" + snv[4] + ";" + snv[5] + ";" + snv[6] + ";" + snv[7] + ";" + snvScaled[0] + ";" + snvScaled[1] + ";" + snvScaled[2] + ";" + snvScaled[3] + ";" + snvScaled[4] + ";" + snvScaled[5] + ";" + snvScaled[6] + ";" + snvScaled[7] + "\"}";

  // // Comment the previous line and uncomment the next line to publish temperature readings in Fahrenheit
  // /*String jsonObject = String("{\"value1\":\"") + (1.8 * bme.readTemperature() + 32) + "\",\"value2\":\""
  //                     + (bme.readPressure()/100.0F) + "\",\"value3\":\"" + bme.readHumidity() + "\"}";*/

  // client.println(String("POST ") + resource + " HTTP/1.1");
  // client.println(String("Host: ") + server);
  // client.println("Connection: close\r\nContent-Type: application/json");
  // client.print("Content-Length: ");
  // client.println(jsonObject.length());
  // client.println();
  // client.println(jsonObject);

  // int timeout = 5 * 10; // 5 seconds
  // while (!!!client.available() && (timeout-- > 0))
  // {
  //   delay(100);
  // }
  // if (!!!client.available())
  // {
  //   Serial.println("No response...");
  // }
  // while (client.available())
  // {
  //   Serial.write(client.read());
  // }

  // Serial.println("\nclosing connection");
  // client.stop();
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

  /////////////////////////screen/////////////////////////////

  u8g2.setI2CAddress(0x7A);
  u8g2.begin();
  /////////////////////////ADC/////////////////////////////
  if (!nau.begin())
  {
    Serial.println("Failed to find NAU7802");
    u8g2.clearBuffer();               // Clear the internal memory of the display
    u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
    u8g2.drawStr(0, 16, "no ");       // Display "Scan" on the screen
    u8g2.drawStr(0, 36, "NAU7802");   // Display "Scan" on the screen
    u8g2.sendBuffer();
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
    u8g2.clearBuffer();               // Clear the internal memory of the display
    u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
    u8g2.drawStr(0, 16, "no LED");    // Display "Scan" on the screen
    u8g2.drawStr(0, 36, "driver");    // Display "Scan" on the screen
    u8g2.sendBuffer();
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
    u8g2.clearBuffer();               // Clear the internal memory of the display
    u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
    u8g2.drawStr(0, 16, "model");     // Display "Scan" on the screen
    u8g2.drawStr(0, 36, "mismatch");  // Display "Scan" on the screen
    u8g2.sendBuffer();                // Transfer internal memory to the display
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

  Serial.println("wait for button press");
  u8g2.clearBuffer();               // Clear the internal memory of the display
  u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
  u8g2.drawStr(0, 16, "Press to");  // Display "Scan" on the screen
  u8g2.drawStr(0, 36, "Calibrate"); // Display "Scan" on the screen
  u8g2.sendBuffer();                // Transfer internal memory to the display
  while (digitalRead(buttonPin) == HIGH)
  {
    // wait for button press
  }

  unsigned long pressTime = millis();

  while (digitalRead(buttonPin) == LOW)
  {
    // wait for button release
  }

  unsigned long releaseTime = millis();
  unsigned long holdTime = releaseTime - pressTime;
  Serial.println(holdTime);
  performScan(true);
  if (holdTime < shortPressTime)
  {
    // Short press
    Serial.println("normal mode");
    // Put your code for short press here...
    scanMode = false;
  }
  else
  {
    Serial.println("Scan collect mode");
    // Long press
    // Put your code for long press here...
    u8g2.clearBuffer();               // Clear the internal memory of the display
    u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
    u8g2.drawStr(0, 16, "Collect");   // Display "Scan" on the screen
    u8g2.drawStr(0, 36, "Mode");      // Display "Scan" on the screen
    u8g2.sendBuffer();                // Transfer internal memory to the display
    scanMode = true;
    initWifi();
    delay(2000); // Wait for 3 seconds before starting calibration
  }

  u8g2.clearBuffer();               // Clear the internal memory of the display
  u8g2.setFont(u8g2_font_inb16_mf); // Choose a suitable font
  u8g2.drawStr(0, 16, "Press to");  // Display "Scan" on the screen
  u8g2.drawStr(0, 36, "Scan");      // Display "Scan" on the screen
  u8g2.sendBuffer();                // Transfer the internal memory to the display
}

void loop()
{
  if (scanMode) // if it is in scan mode it will collect scans from to train model
  {
    if (digitalRead(buttonPin) == LOW) // wait for button press
    {
      /////////////////////////screen/////////////////////////////
      Serial.println("Start SCAN in 3 seconds");
      performScan(false);
      /////// preprocess data
      sample = preprocess();
      if (sample) // if all is good it will upload the scan
      {
        makeIFTTTRequest();
        u8g2.drawStr(0, 16, "Uploaded!"); // Display "Scan" on the screen
      }
      else // if something is wrong it will end here do nothing and in the next part will display the error
      {
        u8g2.drawStr(0, 16, "Error!"); // Display "Scan" on the screen

      }
      u8g2.sendBuffer();               // Transfer internal memory to the display
      delay(2000);                     // Wait for 2 seconds
      u8g2.clearBuffer();              // Clear the internal memory of the display
      u8g2.drawStr(0, 16, "Press to"); // Display "Scan" on the screen
      u8g2.drawStr(0, 36, "Scan");     // Display "Scan" on the screen
      u8g2.sendBuffer();               // Transfer internal memory to the display
    }
  }
  else // this is the interpretation mode, where it scans and inputs the data to the tensorflow model
  {

    int currentButtonState = digitalRead(buttonPin); // checks the state of the button, if flips states on the button press

    // Check if button state has changed
    if (currentButtonState != buttonState)
    {
      delay(50);                                   // Simple debounce
      currentButtonState = digitalRead(buttonPin); // Read the button state again
      if (currentButtonState != buttonState)
      {
        buttonState = currentButtonState; // Update the button state

        // If the new button state is HIGH, then the button was just pressed
        if (buttonState == LOW)
        {
          isScanning = !isScanning; // Toggle the scanning state
        }
      }
    }
    // Wait for input to start the scan
    // check if the pushbutton is pressed. If it is, the buttonState is HIGH:

    if (isScanning)
    {
      Serial.println("Start SCAN");
      performScan(false);
      /////// preprocess data
      sample = preprocess();
      Serial.println(normalized[7] / scaler);
      if ((normalized[7] / scaler) > (1 - offset) && (normalized[7] / scaler) < (1 + offset))
        {
          Serial.println("No Sample");
          u8g2.drawStr(0, 16, "No Sample"); // Display "Too Bright" on the screen
        }
      
      else // if all is correct it will run the model
      {
        // scale from +2.5 and -2.5
        for (int i = 0; i < 8; i++)
        {
          snvScaled[i] = snvScaled[i] / scaler; // Scale the SNV values to 0-1000 to get more resolution
        }
        ////// Run TensorFlow inference
        for (int i = 0; i < 8; i++)
        {
          tflInputTensor->data.f[i] = snvScaled[i]; // Set input tensor values with SNV-transformed data
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

        // Loop through plastic types
        for (int i = 0; i < NUM_PLASTICS; i++)
        {
          Serial.print(PLASTICS[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 3); // Print the likelihood with 6 decimal places

          // Check if the current plastic has a higher likelihood
          if (tflOutputTensor->data.f[i] > maxLikelihood)
          {
            maxLikelihood = tflOutputTensor->data.f[i];
            maxLikelihoodIndex = i;
          }
        }

        if (maxLikelihoodIndex != -1)
        {
          // If the most likely plastic is the same as the last one
          if (maxLikelihoodIndex == lastLikelihoodIndex)
          {
            // Increase the consecutive count
            consecutiveCount++;
          }
          else
          {
            // Reset the consecutive count and update the last likelihood index
            consecutiveCount = 0;
            lastLikelihoodIndex = maxLikelihoodIndex;
            Serial.println("Thinking");
            u8g2.drawStr(0, 16, "Thinking"); // Display "Too Dark" on the screen
          }

          // If the most likely plastic has been detected twice in a row
          if (consecutiveCount >= 2)
          {
            if (maxLikelihood < 0.6)
            {
              // Print the most likely plastic type
              Serial.print("maybe it is: ");
              Serial.println(PLASTICS[maxLikelihoodIndex]);

              u8g2.drawStr(0, 16, "Thinking"); // Display "Too Dark" on the screen
            }
            else
            {
              // Print the most likely plastic type
              Serial.print("Most likely plastic: ");
              Serial.println(PLASTICS[maxLikelihoodIndex]);

              // Display the most likely plastic type and its likelihood on the screen
              u8g2.drawStr(0, 32, PLASTICS[maxLikelihoodIndex]);
              u8g2.setCursor(72, 32);
              u8g2.print(int(tflOutputTensor->data.f[maxLikelihoodIndex] * 100)); // Display likelihood as a percentage
              u8g2.print("%");
              delay(2000); 
            }
          }
        }
      }


      int measuredvbat = analogReadMilliVolts(VBATPIN);
      int batteryVoltage = map(measuredvbat, 1500, 2100, 0, 100); // Map the voltage to a percentage (0-100%)
      u8g2.setCursor(70, 60);
      u8g2.setFont(u8g2_font_scrum_te);
      u8g2.print("Bat:");
      u8g2.print(batteryVoltage); // Display likelihood as a percentage
      u8g2.print("%");
      u8g2.setCursor(-2, 56);
      if (update)
      {
        u8g2.print(".");
      }
      update = !update;
      u8g2.setFont(u8g2_font_inb16_mf);
      u8g2.sendBuffer(); // Transfer the internal memory to the display

      Serial.println("done");
      Serial.println();
    }
    else
    {
      u8g2.clearBuffer();              // Clear the internal memory of the display
      u8g2.drawStr(0, 16, "Press to"); // Display "Scan" on the screen
      u8g2.drawStr(0, 36, "Scan");     // Display "Scan" on the screen
      u8g2.sendBuffer();               // Transfer the internal memory to the display
    }
  }
}
