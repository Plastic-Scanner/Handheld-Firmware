# ESP32-TensorFlow

This is the hardware/firmware/software repository for an ESP32 classifier of the Plastic Scanner project. More technical information is in the [docs](docs.plasticscanner.com) and general information about project on our [website](plasticscanner.com).

## PlatformIO
Requires **PlatformIO**, a cross-platform embedded development toolset. See installation instructions [here ](https://platformio.org/install/)it can be as easy as one-click-plugin-installation (PlatformIO IDE).

In order to build and upload the firmware to DB2.x, connect the board to computer and find the *Build* and *Upload* buttons in the PlatformIO IDE OR use the following commands:

```
$ pio run -t upload 		# build and upload fw image
```

The compiled firmware image is placed in `.pio/build/<board>/firmware.hex`.  
Compilation options can be tweaked in *platformio.ini* file (see [build options](https://docs.platformio.org/en/latest/projectconf/section_env_build.html)).

## WOKWI
Wokwi is a simulator for Arduino and other electronics. It is a great tool for testing and debugging your code before uploading it to the real hardware. It is also a great way to share your projects with others.

How to use wokwi: 
- Enable the wokwi plugin in vsCode, 
- Press F1 and run the command ```wokwi: start simulation```  

## Interactive Python Notebook
The interactive python notebook is a great way to test the classifier and see how it works. It is also a great way to test the classifier on your own data. you can run it by yourself, or use the [Google Colab](https://colab.researh.google.com) version. The link is here: https://colab.research.google.com/drive/1wfOuVHbrcoFD7YLNErialoZrMzzZKGtq#scrollTo=3LlwY8B_h8rJ

## Collect scans for training
To activate the collection mode, in setup when the screen prompts to "press to calibrate" hold the button for 1 second or more, this will enter the ScanCollect part of the sketch that connects the device to the internet and make an IFTTT request to start the collection. You need to setup IFTTT (you can follow [this guide](https://randomnerdtutorials.com/esp32-esp8266-publish-sensor-readings-to-google-sheets/)) and your need to add your details to your own secrets.h file.