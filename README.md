# Guiding-Gaze

## Concept
Aiming to empower visually impaired individuals, we propose a real-time navigation system via a mobile application and wired communication between electronic components.

## Setup
- Install the [DroidCam App](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&pcampaignid=web_share) app on your android.
- Install the [PC version](https://www.dev47apps.com/droidcam/windows/) of the app on your laptop. Similar versions are available for MacOS and linux.
- Follow [this guide](https://www.dev47apps.com/droidcam/connect/) to get it running. Once working, you can use your phone as a webcam.
- There is an option for WiFi connection as well, but due to high computational requirements of the model, we have used a data cable to reduce lag.
- Create a virtual environment for guiding gaze:
```
pip install virtualenv
```
```
virtualenv -p /usr/bin/python3 guiding-gaze
source guiding-gaze/bin/activate
```
- Install Guiding Gaze on your PC:
```
  git clone https://github.com/MananGaur19/Guiding-Gaze.git
```
- Go inside the repo and install the dependencies

```
pip install -r requirements.txt
```
  
![Architecture](https://github.com/MananGaur19/Guiding-Gaze/assets/56295289/0b1577e4-9132-46ad-acb2-4b1d376f26f7)

## Modules
### Depth Estimation
- We use MiDAS to calculate object distance from the user.
- Run the `MIDAS/MIDAS.ipynb` file to run this on your machine.
![image](https://github.com/MananGaur19/Guiding-Gaze/assets/56295289/2b5da6d2-0da0-49d4-882c-184f93168118)

### Scene Recognition
Run the `Scene Recognition/scene_recognition.ipynb` on your PC for scene recognition.

### Obstruction Detection
Run the `Obstruction/wall_obstruction.ipynb` on your PC for detecting obstructions.
![image](https://github.com/MananGaur19/Guiding-Gaze/assets/56295289/f6762794-1b98-4e9f-84a7-de74a7514b83)

### Face Recognition
You can run face recognition via the file: `Face Recognition/face_recognition.ipynb`
<img width="452" alt="image" src="https://github.com/MananGaur19/Guiding-Gaze/assets/56295289/cab94169-7921-4dde-8ef3-bca258b6b834">

## Running Guiding Gaze













