# SailSwarm Obstacle Detection

## Project Overview
The Sailswarm project aims to use a swarm of small autonomous sailboats to collect environmental data on Lake Constance. These data collection missions will last for up to a week, therefore scene awareness and survivability are critical. This module integrates three sensors that can be used independently and together throughout day and night and various weather conditions to detect obstacles on the lake such as other boats, people, ducks, buoys, and more.

## Hardware Components

![alt text](https://github.com/margaretmichelle/SailSwarm-ObstacleDetection/blob/main/images/box.JPG?raw=true)
![alt text](https://github.com/margaretmichelle/SailSwarm-ObstacleDetection/blob/main/images/sensors.JPG?raw=true)

The obstacle detection module contains the following:

+ **Raspberry Pi 4:** Serving as the main processing unit, responsible for controlling the sensors and performing iage and radar data processing.

+ **RGB Fisheye Camera:** Raspberry Pi compatible camera to collect valuable visible light data (https://www.amazon.de/Raspberry-Pi-Kamera-Weitwinkel-Nachtsicht-Kameramodul-Raspberry/dp/B0748KF97S?language=en_GB).

+ **FLIR Lepton 3.0 Thermal Camera:** FLIR Lepton 3.0 thermal camera to analyze scenes using long wave infrared radiation and especially for night and low visibility scenes (https://www.sparkfun.com/purethermal-mini-pro-jst-sr-with-flir-lepton-3-5.html).

+ **AWR1843BOOST mmWave Sensor:** mmWave radar sensor used to detect obstacles and determine distances (https://www.ti.com/tool/AWR1843BOOST).

## Software Components

The codebase is written in Python and includes the following:

### Data Collection: 
Scripts to run on the Raspberry Pi to collect raw data for testing processing pipelines, both isolated to the Raspberry Pi and with a socket connection to a PC using the Raspberry Pi's Wi-Fi Access Point.

### Sensor Data Processing:
Scripts used to calibrate and process the sensor outputs as well as final sensor fusion data processing script.

### Sensor Mount CAD:
CAD files for the 3D printed mount the sensors and Raspberry Pi sit on inside the waterproof enclosure.


## Contact Information:
+ Primary Author: Margaret Lee
+ Project Lead: Pranav Kedia
+ Institution: University of Konstanz, Center for the Advanced Study of Collective Behavior. Internship and funding through DAAD Rise and MITACS.