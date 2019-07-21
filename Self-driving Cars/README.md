This folder contains code from the course [State Estimation and Localization for Self-Driving Cars](https://www.coursera.org/learn/state-estimation-localization-self-driving-cars/home/welcome) that can be found on Coursera. 

The following things can be found here:

- **carMeasuringAngle.py** An example that is brought up during the lectures where a car is driving towards a landmark at a known location (*D*) along the x-axis. While measuring the angle to the top (*S*) of the landmark. The objective is to use this measurement to estimate the cars position along the x-axis. This has been done both with EKF and UKF. 
    
- **EKF.py**: A simple implementation of the Extended Kalman Filter. This is not a general implementation, it is only implemented to fit the above mentioned example. 
    
- **UKF.py**: A simple implementation of the Unscented Kalman Filter. This is a general implementation as it does not make any assumptions that are specific to the example above. 
