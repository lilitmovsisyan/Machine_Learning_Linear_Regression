# -*- coding: utf-8 -*-
"""
LILIT_MOVSISYAN_LinearRegression.py

This script contains a function that runs a linear regression model 
over randomly generated data, prints the results and plots the generated model, 
as well as a function to help with plotting comparisons of different models.

Please see LILIT_MOVSISYAN_run_lr.py for the implementation of this 
linear regression function and further plots.

  ----------------------------------------------------------------------
  author:       Lilit Movsisyan
  Date:         01/08/2019
  ----------------------------------------------------------------------
"""
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'




def run_linear_regression(
        gradient=1, 
        intercept=0, 
        training_epochs=100, 
        learning_rate=0.005, 
        Ngen=100, min_x=0, 
        max_x=1, 
        noise=0.1, 
        verbose=True,
        plotting=True,
):

    """
FUNCTION SUMMARY:

This function runs a tensorflow linear regression algorithm 
on a set of randomly generated, noisy, linearly correllated data.

This function takes the following arguments with these given defaults:
    gradient=1,          - The target gradient of the linear relationship to be modelled
    intercept=0,         - The target intercept of the linear relationship to be modelled
    training_epochs=100, - Number of training epochs to run the optimiser over
    learning_rate=0.005, - Step size (how much to adjust the estimated parameters by each training epoch)
    Ngen=100,            - Number of random sample data points
    min_x=0,             - maximum data_x value to be randomly generated
    max_x=1,             - minimum data_x value to be randomly generated
    noise=0.1,           - rate of noise in data
    verbose=True,        - when True, verbose print statements will be enabled
    plotting=True        - when True, plots the generated model against data

This function will print out a summary of the optimisation, including the final estimated values
for the gradient and intercept parameters, and the total loss they generate.

The function returns a dictionary containing the following:
    
    "epoch": epoch                       - a list containing indices of training epochs
    "chi2": chi2                         - a list of the total loss indexed by training_epoch
    "estimated_c_list": estimated_c_list - a list of the estimated intercept parameter, c, for each training_epoch
    "estimated_m_list": estimated_m_list - a list of the estimated gradient parameter, m, for each training_epoch

--------------------------------------------------------------------------------------------------
FURTHER FUNCTION EXPLANATION:

First, a set of random numbers, data_x, is generated to represent the x feature in the relationship.
The function data_y = gradient*data_x + intercept (+ some noise) is applied 
to each member of data_x, where the gradient and intercept are defined by the 
gradient and intercept arguments passed to the function. A randomised noise value is added to each 
datapoint to simulate real-world data.

Then, the linear regression algorithm is applied to this set of data_x and data_y values 
to provide estimates for the values of the gradient parameter, m and the intercept parameter, c,
needed to model this function. 
The tensorflow GradientDescentOptimizer evaluates the loss of the model function's 
predicted y values against the corresponding data_y values and updates the estimated m and c parameters.
Loss is defined as the chi squared sum (sum of squared differences).
The estimated parameters are updated by learning_rate over a number of training_epochs.
    """

    # print introductory statement:
    print("Using linear regression to model data.")
    print("Target gradient = ", gradient)
    print("Target intercept = ", intercept)
    print("......................................")

    ##############################################################################
    # DEFINE GRAPH:
    ##############################################################################
    
    # tf Graph input:
    #  x_: is the tensor for the input data (the placeholder entry None is used for that;
    #     and the number of features input (n_input = 1).
    #
    #  y_: is the tensor input data computed from the generated data x values
    #
    #
    #  y: is the tensor for the output value of the function that is being approximated by 
    #     the the model, calculated from y = mx+c
    x_ = tf.placeholder(tf.float32, [None, 1], name="x_")
    y_ = tf.placeholder(tf.float32, [None, 1], name="y_")
    
    # parameters of the model are m (gradient) and c (constant offset, intercept)
    #   - pick random starting values for fit convergence
    c = tf.Variable(tf.random_uniform([1]), name="c")
    m = tf.Variable(tf.random_uniform([1]), name="m")
    y = m * x_ + c
    
    # assume all data have equal uncertainties to avoid having to define an example by
    # example uncertainty, and define the loss function as a simplified chi^2 sum
    loss = tf.reduce_sum((y - y_) * (y - y_))
    
    # define a gradient descent optimiser
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # prepare data - data_x and data_y are np arrays that will get fed into the optimisation
    data_x = (max_x-min_x)*np.random.random([Ngen, 1])+min_x
    data_y = np.zeros([Ngen,1])
    for i in range(Ngen):
        data_y[i] = gradient*data_x[i]+intercept + noise*np.random.randn()

    ##############################################################################
    # START SESSION:
    ##############################################################################     
    
    # prepare the session for evaluation and initialise the variables.
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # print("check random seed: c ", sess.run(c)) ### Doesn't seed random values. Need to look into this later.
    # print("check random seed: m ", sess.run(m))
    # print("check random seed: data_x ", data_x[0:5])

    ##############################################################################
    # RUN OPTIMISATION:
    ############################################################################## 
    
    # epoch and chi2 are the training step and the chi2 sum (neglecting the error normalisation)
    # for the problem.  These will be logged to inspect the fit convergence obtained.
    epoch = []
    chi2  = []
    estimated_c_list  = []
    estimated_m_list  = []
    if verbose:
        print("Epoch   m         c") 
    for step in range(training_epochs):
        # run the minimiser
        sess.run(train_step, feed_dict={x_: data_x, y_: data_y})
    
        # log the data and print output
        epoch.append(step)
        chi2.append(sess.run(loss, feed_dict={x_: data_x, y_: data_y}))
        estimated_c_list.append(sess.run(c))
        estimated_m_list.append(sess.run(m))
        
        if verbose:
            print(step, sess.run(m), sess.run(c), chi2[step])

    ##############################################################################
    # PRINT RESULTS AND PLOT MODEL:
    ############################################################################## 

    results = {
            "epoch": epoch,
            "chi2": chi2, 
            "estimated_c_list": estimated_c_list,
            "estimated_m_list": estimated_m_list, 
             "data_x": data_x, 
             "data_y": data_y, 
            # "optimal_c": sess.run(c), 
            # "optimal_m": sess.run(m)
            }
        
    # print optimisation results
    print("Optimisation process has finished. The optimal values of parameters are:")
    print("  m = ", sess.run(m), " [input value = ", gradient, "]" )
    print("  c = ", sess.run(c), " [input value = ", intercept, "]" )
    print("  loss function for the optimal parameters = ", chi2[training_epochs-1])
    print("______________________________________")
    
    # plot graph of data against model
    if plotting:
        plt.plot(data_x, data_y, 'r.')
        plt.plot([min_x, max_x], [sess.run(c), sess.run(m)*max_x+sess.run(c)], 'b-')
        plt.ylabel('f(x)')
        plt.xlabel('x')
        plt.legend(['Data', 'Model'], loc=0)
        plt.title('Linear Regression Example: target gradient={0}, target intercept={1}'.format(gradient, intercept))
        plt.show()
    
    
    return results
    
    # return (epoch, chi2, estimated_c_list, estimated_m_list, data_x, data_y, sess.run(c), sess.run(m))

######################################################################################


def plot_comparison(results_set, comparison):
    """
    This function plots either the loss, estimated c parameter, or estimated m parameter as a function of training epochs for 
    This function takes the following input arguments:

     - results_set - a list of results generated by run_linear_regression() function calls. 
                    i.e. each result in the set is a dictionary containing arrays of values
                    (as produced by the run_linear_regression() function above) for that application of the function.

     - comparison  - a string equal to either 'loss', 'c', or 'm' indicating the axis along which to make comparisons.
    """
    if comparison == 'loss':
        variable = "chi2"
        y_label  = "loss"
        title    = "Linear Regression: Change in loss"
        filename = "activity2_loss.pdf"
    elif comparison == 'c':
        variable = "estimated_c_list"
        y_label  = "Intercept parameter"
        title    = "Linear Regression: Change in parameter c"
        filename = "activity2_estimated_c.pdf"
    elif comparison == 'm':
        variable = "estimated_m_list"
        y_label  = "Gradient parameter"
        title    = "Linear Regression: Change in parameter m"
        filename = "activity2_estimated_m.pdf"
    
    # Plot the relationships:
    for result in results_set:
        plt.plot(result["epoch"], result[variable])

    # Labels:
    plt.semilogy()
    plt.ylabel(y_label)
    plt.xlabel('Training Epoch')
    plt.title(title)
    
    # Define legend:
    plt.legend(["m=1", "m=10", "m=100"], loc=0) # Haven't had time to make the legend flexible, so currently hardcoded to the values for activity 2.
    
    # Set plot settings:
    plt.tight_layout()
    plt.grid(True)
    
    # Save figure:
    plt.savefig(filename)

    plt.show()