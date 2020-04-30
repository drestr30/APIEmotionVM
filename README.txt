Vagrant VM with Ubuntu 16.04

Installed packages:

- Python 3.7.0
- TensorFlow 1.15.0
- Keras 2.2
- Flask
- OpenCV
  ...and dependences

Emotion detection service is itinialized by running in vagrant shell:

python code/emotion_app.py

The app will run at local port 88888
The request should contain the encoded image


/code: contains the code for running the emotion service API.
	emotion_app.py: flask implementation of the server API.
	load_model.py: Keras and tf implementation of DL models.
	/models: emotion and facial detection models. 

client.py: client example with realtime webcam implementation for emotion detection
	   using simple html request, Imutils videos and OpenCV. 