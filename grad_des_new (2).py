#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Function to compute the value of the quadratic function
def func(x, y):
    return x**2 + y**2

# Function to compute the gradients of the quadratic function
def grad(x, y):
    return np.array([2*x, 2*y])

# Gradient Descent Algorithm
def gradient_descent(starting_point, learning_rate, num_iterations):
    x_history = [starting_point[0]]
    y_history = [starting_point[1]]
    current_point = np.array(starting_point)
    
    for _ in range(num_iterations):
        gradient = grad(*current_point)
        current_point = current_point - learning_rate * gradient
        x_history.append(current_point[0])
        y_history.append(current_point[1])
    
    return x_history, y_history

# Streamlit application
st.title('Interactive Gradient Descent Visualization (2D Quadratic Function)')

# User inputs
starting_x = st.slider('Starting Point X', -10.0, 10.0, 0.0)
starting_y = st.slider('Starting Point Y', -10.0, 10.0, 0.0)
learning_rate = st.slider('Learning Rate', 0.001, 1.0, 0.1)
num_iterations = st.slider('Number of Iterations', 1, 100, 10)

# Compute gradient descent
starting_point = (starting_x, starting_y)
x_history, y_history = gradient_descent(starting_point, learning_rate, num_iterations)
x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = func(X, Y)

# Create Plotly figure
fig = go.Figure()

# Add surface plot
fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, showscale=True, name='Surface'))

# Add gradient descent path
fig.add_trace(go.Scatter3d(
    x=x_history, 
    y=y_history, 
    z=[func(x, y) for x, y in zip(x_history, y_history)],
    mode='lines+markers', 
    marker=dict(color='red', size=5, line=dict(color='black', width=1)),
    line=dict(color='red', width=4),
    name='Gradient Descent Path'
))

# Add annotations
for i in range(len(x_history)):
    fig.add_trace(go.Scatter3d(
        x=[x_history[i]], 
        y=[y_history[i]], 
        z=[func(x_history[i], y_history[i])],
        mode='markers+text',
        marker=dict(color='blue', size=7),
        text=[f'({x_history[i]:.1f}, {y_history[i]:.1f})'],
        textposition='top center',
        showlegend=False
    ))

# Update layout for better visibility
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='f(x, y)',
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10]),
        zaxis=dict(range=[0, func(0, 0)]),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    title='Gradient Descent on f(x, y) = x^2 + y^2',
    margin=dict(l=0, r=0, b=0, t=40)
)

# Display plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

