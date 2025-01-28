#!/usr/bin/env python3
"""
"""

import numpy as np
import os
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
import sys
from warnings import simplefilter

try:
    __projectdir__ = Path(os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/'))
except NameError:
    __projectdir__ = Path(os.path.abspath(""))


import copy
import datetime
import numpy as np
import pandas as pd
import pickle

def dorigobonhet(df, yvar, xvar, yvarf = None, xvarf = None, dff = None, nbootstrap = 1000, printdetails = False, returnmodel = False):
    if dff is None:
        dff = df
    if yvarf is None:
        yvarf = yvar.split('_')[0] + 'a' + '_' + '_'.join(yvar.split('_')[1: ])
    if xvarf is None:
        xvarf = xvar.split('_')[0] + 'a' + '_' + '_'.join(xvar.split('_')[1: ])

    cov = df[[yvar, xvar]].cov().iloc[0, 1]
    var = df[[yvar, xvar]].cov().iloc[1, 1]
    covf = dff[[yvarf, xvarf]].cov().iloc[0, 1]
    varf = dff[[yvarf, xvarf]].cov().iloc[1, 1]

    betahat = (cov - covf) / (var - varf)

    if printdetails is True:

        print('Covariance matrix real:')
        print(df[[yvar, xvar]].cov())
        
        print('Covariance matrix fake:')
        print(dff[[yvarf, xvarf]].cov())
        
        # get basic estimates
        model = smf.ols(formula = yvar + ' ~ ' + xvar, data = df).fit()
        print('Basic estimate: ' + str(model.params[xvar]))

        # estimate from fake data
        model = smf.ols(formula = yvarf + ' ~ ' + xvarf, data = dff).fit()
        print('Fake estimate: ' + str(model.params[xvarf]))

        print('Heteroskedasticity estimate: ' + str(betahat))


    # BOOTSTRAP BASIC

    # keep only relevant variables
    df2 = df[[yvar, xvar]].dropna().copy()
    dff2 = dff[[yvarf, xvarf]].dropna().copy()

    # take draws of dataset and get estimates
    betahatsample = []
    i = 0
    itry = 0
    while i < nbootstrap:
        # get dataset the same length as df3
        dfs = df2.sample(n = len(df2), replace = True)
        dfsf = dff2.sample(n = len(dff2), replace = True)

        # now get same estimates of betahat
        cov = dfs[[yvar, xvar]].cov().iloc[0, 1]
        var = dfs[[yvar, xvar]].cov().iloc[1, 1]
        covf = dfsf[[yvarf, xvarf]].cov().iloc[0, 1]
        varf = dfsf[[yvarf, xvarf]].cov().iloc[1, 1]

        betahatboot = (cov - covf) / (var - varf)

        if pd.isnull(betahatboot):
            itry += 1
            if itry > nbootstrap:
                raise ValueError(str(nbootstrap) + ' failures while running bootstrap estimates. Maybe no data available.')
            continue
        else:
            betahatsample.append(betahatboot)
            i+=1

    if returnmodel is False:
        return(betahat, betahatsample)
    else:
        class Model(object):
            pass

        model = Model()

        model.params = pd.Series([betahat], index = [xvar])

        std = np.std(betahatsample)
        model.bse = pd.Series([std], index = [xvar])

        # count number of betahats either side of zero
        negbetahatsample = len([betahat for betahat in betahatsample if betahat <= 0])
        posbetahatsample = len([betahat for betahat in betahatsample if betahat >= 0])
        pval = min([negbetahatsample, posbetahatsample]) / len(betahatsample)
        model.pvalues = pd.Series([pval], index = [xvar])

        model.nobs = None
        model.rsquared = None

        model.betahatsample = betahatsample

        return(model)


