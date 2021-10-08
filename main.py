#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:13:45 2021

@author: vgomez
"""

from scipy.stats import truncnorm 
import numpy as np 
import matplotlib.pyplot as plt 

# we consider x-axis normalized between 0 and 1
def plot_model(loc_ar=.3, scale_ar=.05, loc_de=.8, scale_de=.1):
    # arrivals
    a_ar = -loc_ar/scale_ar
    b_ar = (1-loc_ar)/scale_ar

    # departures
    a_de = -loc_de/scale_de
    b_de = (1-loc_de)/scale_de

    x = np.linspace(-.1, 1.1, num=100)
    pdf_ar = truncnorm.pdf(x, a_ar, b_ar, loc=loc_ar, scale=scale_ar)
    pdf_de = truncnorm.pdf(x, a_de, b_de, loc=loc_de, scale=scale_de)
    cdf_ar = truncnorm.cdf(x, a_ar, b_ar, loc=loc_ar, scale=scale_ar)
    cdf_de = truncnorm.cdf(x, a_de, b_de, loc=loc_de, scale=scale_de)

    fig, ax = plt.subplots(2)
    ax[0].plot(x, pdf_ar , '-b')
    ax[0].plot(x, pdf_de, '-r')
    ax[0].set_title('pdfs')

    ax[1].plot(x, cdf_ar , '--b')
    ax[1].plot(x, cdf_de, '--r')
    ax[1].plot(x, cdf_ar-cdf_de, 'r')
    ax[1].set_title('cdfs')



plot_model()