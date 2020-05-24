# Neuro Trust Region
Implementation of the paper: https://arxiv.org/pdf/2004.09058.pdf   
The algorithm has two main steps:   

## 1. Description
### 1.1 Finding the interpolation of the black-box function using a neural network

Below, you can an interpolation of the 2D-Rosenbrock function, starting around point (0.5, 0.5) with delta (radius) of 0.5.  
![image](https://user-images.githubusercontent.com/8312051/82767391-19d55a00-9ddc-11ea-8681-966583877e90.png)

### 1.2 Choosing the best candidate point in the region 

The second step is to use the interpolation function and choose a point in the region: 
![image](https://user-images.githubusercontent.com/8312051/82767514-17273480-9ddd-11ea-9f12-b845dfbc7be3.png)



## 2. How to Setup \& Run the code

### 2.1 Setup
You can run the code by installing packing dependencies. Simply run:   
```
pip install -r requirements.txt
```   

The depenencies are PyTorch (1.5+) for automatic differentiation, Matplotlib for visualization, and torch optimizer for implementation of other optimizers for PyTorch.

Once installed, simply run:
```
python main.py
```
You can change the solver parameters by modifiying settings of ``SolverParams()`` class in `problem.py`.
