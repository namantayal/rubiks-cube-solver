# Rubiks Cube Solver

It is a 3x3 Rubik's cube solver which utilizes the python OpenCV library for image processing.

The workflow of the project is-
1. Plots a combination of HSV masks to detect the cube.
2. Divids the cube into cubelets
3. Detect the color of each cubelet
4. Obtains the solution
5. Plots the solution sequentially with checks to validate the moves




## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install [OpenCV](https://pypi.org/project/opencv-python/) and [kociemba](https://pypi.org/project/kociemba/) (package to obtain cube solution).

```bash
pip install opencv-python
pip install kociemba
```

> **WARNING**: Visual studio C++ build tools are required for kociemba package to work.


## Usage

Modify to 0 for internal webcam and 1 for an external webcam
```
cap = cv.VideoCapture(1)
```

Modify HSV values according to your cube for color detection
```
colors = np.array([[135, 115, 250],
                   [50, 215, 135],
                   [215, 190, 90],
                   [40, 220, 219],
                   [210, 210, 210],
                   [90, 145, 255]])  # r g b y w o
```

Modify the HSV range used for creating a mask according to your cube
```
#Blue Color
mask_b = cv.inRange(blur, np.array([74, 81, 178]), np.array([112, 255, 255]))
#Green Color
mask_g = cv.inRange(blur, np.array([40, 128, 180]), np.array([78, 255, 255]))
#Yellow Color
mask_y = cv.inRange(blur, np.array([21, 159, 167]), np.array([40, 235, 255]))
#Orange Color
mask_o = cv.inRange(blur, np.array([0, 120, 190]), np.array([20, 186, 255]))
#White Color
mask_w = cv.inRange(blur, np.array([0, 0, 190]),np.array([130,39,255]))
#Red Color
mask_r = cv.inRange(blur, np.array([165, 48, 213]), np.array([179, 170, 255]))
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
