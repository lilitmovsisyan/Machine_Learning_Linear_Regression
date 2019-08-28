# -*- coding: utf-8 -*-
"""
LILIT_MOVSISYAN_run_lr.py

This scipt runs linear regression functions defined in LILIT_MOVSISYAN_LinearRegression.py
and plots graphs to investigate them.

Activity 1: run linear regression function to model data with gradient=2 and intercept=0
and plot graphs to investigate convergence behaviour of the model.

Activity 2: run linear regression functions to model data with intercept=0 and the following gradients:
m=1, m=10 and m=10. Plot graphs to investigate convergence behaviour of the model.

  ----------------------------------------------------------------------
  author:       Lilit Movsisyan
  Date:         01/08/2019
  ----------------------------------------------------------------------
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import LILIT_MOVSISYAN_LinearRegression as lr

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# # set the seed for random number generation to ensure we always get the same data
# np.random.seed(1) # I don't think this is working as planned. Need to look into this another time: control scope.

##############################################################################
# ACTIVITY 1:
############################################################################## 


# LINEAR REGRESSION FOR ACTIVITY 1:
results_1 = lr.run_linear_regression(gradient=2, intercept=0.5, training_epochs=300, verbose=False, plotting=True)


############################################################################## 

# MAIN PLOTS FOR ACTIVITY 1:

plt.plot(results_1["epoch"], results_1["chi2"])
plt.ylabel('Loss')
plt.xlabel('Training epoch')
plt.title('Linear Regression: Change in loss over 300 epochs')
plt.show()

plt.plot(results_1["epoch"], results_1["estimated_c_list"], 'r')
plt.ylabel('c')
plt.xlabel('Training epoch')
plt.title('Linear Regression: Change in c (intercept) over 300 epochs')
plt.show()

plt.plot(results_1["epoch"], results_1["estimated_m_list"], 'g')
plt.ylabel('m')
plt.xlabel('Training epoch')
plt.title('Linear Regression: Change in m (gradient) over 300 epochs')
plt.show()

###############################################################################

# SOME FURTHER PLOTS (to zoom in):

print("\nACTIVITY\n")
print("From looking at verbose printout above, we see stabilisation of values m, c, and loss to 7 decimal places happens around epoch 200~210")

# loss

plt.figure(figsize=(18,3))
plt.suptitle('Converging to a stable value of cost (chi^2 loss function): Zooming in to training epochs for detail', va='top')
plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.8, bottom=0.3)

plt.subplot(1,3,1)
plt.plot(results_1["epoch"], results_1["chi2"])
plt.ylabel('Loss')
plt.xlabel('Training epoch')
plt.title('From epoch 0 to 300:')

plt.subplot(1,3,2)
plt.plot(results_1["epoch"][100:250], results_1["chi2"][100:250])
plt.ylabel('Loss')
plt.xlabel('Training epoch')
plt.title('From epoch 100 to 250:')

plt.subplot(1,3,3)
plt.plot(results_1["epoch"][150:250], results_1["chi2"][150:250])
plt.ylabel('Loss')
plt.xlabel('Training epoch')
plt.title('From epoch 150 to 250:')

plt.savefig('activity1_loss.pdf')
plt.show()

# OPTIONAL EXTRA PLOTS FOR ESTIMATED GRADIENT AND INTERCEPT:
# c

plt.figure(figsize=(18,3))
plt.suptitle('Converging to a stable value of c (intercept): Zooming in to training epochs for detail', va='top')
plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.8, bottom=0.3)

plt.subplot(1,3,1)
plt.plot(results_1["epoch"], results_1["estimated_c_list"], 'r')
plt.ylabel('c (intercept)')
plt.xlabel('Training epoch')
plt.title('From epoch 0 to 300:')

plt.subplot(1,3,2)
plt.plot(results_1["epoch"][150:250], results_1["estimated_c_list"][150:250], 'r')
plt.ylabel('c (intercept)')
plt.xlabel('Training epoch')
plt.title('From epoch 150 to 250:')

plt.subplot(1,3,3)
plt.plot(results_1["epoch"][180:250], results_1["estimated_c_list"][180:250], 'r')
plt.ylabel('c (intercept)')
plt.xlabel('Training epoch')
plt.title('From epoch 180 to 250:')

plt.savefig('activity1_c_intercept.pdf')
plt.show()


# m

plt.figure(figsize=(18,3))
plt.suptitle('Converging to a stable value of m (gradient): Zooming in to training epochs for detail', va='top')
plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.8, bottom=0.3)

plt.subplot(1,3,1)
plt.plot(results_1["epoch"], results_1["estimated_m_list"], 'g')
plt.ylabel('m (gradient)')
plt.xlabel('Training epoch')
plt.title('From epoch 0 to 300:')

plt.subplot(1,3,2)
plt.plot(results_1["epoch"][150:250], results_1["estimated_m_list"][150:250], 'g')
plt.ylabel('m (gradient)')
plt.xlabel('Training epoch')
plt.title('From epoch 150 to 250:')

plt.subplot(1,3,3)
plt.plot(results_1["epoch"][180:220], results_1["estimated_m_list"][180:220], 'g')
plt.ylabel('m (gradient)')
plt.xlabel('Training epoch')
plt.title('From epoch 200 to 250:')

plt.savefig('activity1_m_gradient.pdf')
plt.show()




###############################################################################
# ACTIVITY 2:
###############################################################################

# LINEAR REGRESSION FOR ACTIVITY 2:
# run the linear regression over the three different cases, and save results in results_set
results_a = lr.run_linear_regression(gradient=1, intercept=0, verbose=False, plotting=True)
results_b = lr.run_linear_regression(gradient=10, intercept=0, verbose=False, plotting=True)
results_c = lr.run_linear_regression(gradient=100, intercept=0, verbose=False, plotting=True)
results_set = [results_a, results_b, results_c]


# PLOTS FOR ACTIVITY 2:

# Use results_set defined above to plot comparison graphs:
lr.plot_comparison(results_set, "loss")
lr.plot_comparison(results_set, "c")
lr.plot_comparison(results_set, "m")