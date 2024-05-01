#!/usr/bin/env python3
"""
"""


import os
from pathlib import Path
import sys

try:
    __projectdir__ = Path(os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/'))
except NameError:
    __projectdir__ = Path(os.path.abspath(""))


import copy
import datetime
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
import numpy as np
import pandas as pd
import pickle

from event_shock_func import *

sys.path.append(str(__projectdir__ / Path('submodules/yield-curve-comp/')))
from yieldcurve_gen_func import *

# Definitions:{{{1
# create two template dicts for each of the rel/shockdicts we need for the functions
# call the first, smaller one by default
# Reldict Daily:{{{2
reldict_daily_basic = {}
reldict_daily_basic['m1c'] = {'rday': -1, 'open': False, 'weekendignore': True}
reldict_daily_basic['1c'] = {'rday': 1, 'open': False, 'weekendignore': True}

reldict_daily_extended = copy.deepcopy(reldict_daily_basic)
reldict_daily_extended['m1o'] = {'rday': -1, 'open': True, 'weekendignore': True}
reldict_daily_extended['1o'] = {'rday': 1, 'open': True, 'weekendignore': True}
# this is to get the change before the event
reldict_daily_extended['m1c_f7'] = {'rday': -1, 'open': False, 'weekendignore': True, 'ffill': 7}
reldict_daily_extended['m90c_f7'] = {'rday': -90, 'open': False, 'weekendignore': True, 'ffill': 7}

# Shockdict Daily:{{{2
shockdict_daily_basic = {}
shockdict_daily_basic["m1c_1c"] = ["m1c", "1c"]

shockdict_daily_extended = {}
shockdict_daily_extended["m1c_1c"] = ["m1c", "1c"]
shockdict_daily_extended["m1c_1o"] = ["m1c", "1o"]
shockdict_daily_extended["m1o_1o"] = ["m1o", "1o"]
shockdict_daily_extended["m90c_m1c"] = ["m90c_f7", "m1c_f7"]

# Reldict Return Daily:{{{2
reldictreturn_daily_extended = {'m1c': 'm1c_f7'}

# Reldict Intraday:{{{2
# reldict to apply as input in function to getdfintrarel_reldict
reldict_intra_basic = {}
# if event is 1600M, this would take the last trade at 1500M-1549M (with 5 minute interval data)
reldict_intra_basic['m1h'] = {'rpos': -3, 'open': False, 'ffill': 9}
# if event is 1600M, this would take the first trade at 1610M-1659M (with 5 minute interval data)
reldict_intra_basic['1h'] = {'rpos': 2, 'open': True, 'bfill': 9}

reldict_intra_extended = copy.deepcopy(reldict_intra_basic)
reldict_intra_extended['m20m'] = {'rpos': -3, 'open': False, 'ffill': 2}
# reldict_intra_extended['m15mp'] = {'rpos': -2, 'open': False, 'ffill': 1}
reldict_intra_extended['m3h'] = {'rpos': -3, 'open': False, 'ffill': 33}
reldict_intra_extended['m6h'] = {'rpos': -3, 'open': False, 'ffill': 69}
reldict_intra_extended['20m'] = {'rpos': 2, 'open': True, 'bfill': 2}
# reldict_intra_extended['15mp'] = {'rpos': 1, 'open': True, 'bfill': 1}
reldict_intra_extended['3h'] = {'rpos': 2, 'open': True, 'bfill': 33}
reldict_intra_extended['6h'] = {'rpos': 2, 'open': True, 'bfill': 69}

# Shockdict Intraday:{{{2
# shockdict to apply as input in function to getbefaft
shockdict_intra_basic = {}
shockdict_intra_basic["m1h_1h"] = ["m1h", "1h"]

shockdict_intra_extended = {}
shockdict_intra_extended["m20m_20m"] = ["m20m", "20m"]
shockdict_intra_extended["m1h_1h"] = ["m1h", "1h"]
shockdict_intra_extended["m3h_3h"] = ["m3h", "3h"]
shockdict_intra_extended["m6h_6h"] = ["m6h", "6h"]

# Auxilliary Functions:{{{1
def processdates(dates, zonestodo, zonesavailable):
    """
    I can either input a dates: {'AUS' ['20100101d'], .'CAN': ['20110202d']}
    Or I can input dates = ['20100101d', '20110202d'] and zonestodo = ['AUS', CAN'] in which case this function generates a dates {'AUS': ['20100101d', '20110202d'], 'CAN': ['20100101d', '20110202d']}
    Finally, I can also specify dates = ['20100101d', '20110202d'] and zonestodo = 'all' in which case this function allows me to get a dates with the list for every zone for which data is available

    Note this function also works identically with a minutedict rather than a dates
    """
    if zonestodo is None:
        if not isinstance(dates, dict):
            raise ValueError('If zonestodo is None then dates should be a dict.')
    else:
        if not isinstance(dates, list):
            raise ValueError('If zonestodo is not None then dates should be a list.')

        if zonestodo == 'all':
            zonestodo = zonesavailable

        dates = {zone: dates for zone in zonestodo}

    return(dates)


def getdatedictfromdf(df, datevar, dropna = True):
    """
    Should input a dataframe with 'zone' and datevar as variables
    For example df['zone'] = ['AUS', 'AUS', 'CAN'], df[datevar] = ['20100101d', '20110202d', '20120303d']
    Then datedict = {'AUS': ['20100101d', '20110202d'], 'CAN': ['20120303d']}
    """
    # drop any nan dates
    df = df[df[datevar].notna()].copy()

    # sort df by datevar
    df = df.sort_values(datevar)

    zones = list(df['zone'])
    dates = list(df[datevar])
    datedict = {}
    for i in range(len(zones)):
        zone = zones[i]
        date = dates[i]
        if zone not in datedict:
            datedict[zone] = []
        # remove duplicates
        if date not in datedict[zone]:
            datedict[zone].append(date)

    return(datedict)


# Merge Functions:{{{1
def mergezonedir(mergedir, picklefiles = False):
    zonefiles = os.listdir(mergedir)
    if picklefiles is True:
        zonefiles = [zonefile for zonefile in zonefiles if zonefile.endswith('.pickle')]
    else:
        zonefiles = [zonefile for zonefile in zonefiles if zonefile.endswith('.csv')]
    zonefiles = sorted(zonefiles)

    # read all files
    dfzonelist = []
    for zonefile in zonefiles:
        if picklefiles is True:
            with open(mergedir / Path(zonefile), 'rb') as f:
                dfzone = pickle.load(f)
        else:
            dfzone = pd.read_csv(mergedir / Path(zonefile), index_col = [0, 1])

        dfzonelist.append(dfzone)

    # merge verically
    df = pd.concat(dfzonelist)

    return(df)

        
# Bond-Specific Functions:{{{1
def getsinglemat(mat):
    """
    This should be something like y04 or m03
    """
    mat_letter = mat[0]
    mat = int(mat[1: ])

    if mat_letter in ['y', 'z']:
        None
    elif mat_letter == 'm':
        mat = mat / 12
    elif mat_letter == 'd':
        mat = mat / 365
    else:
        raise ValueError('mat misspecified: ' + mat + '.')

    return(mat)


def getridgemats(ridgerange):
    """
    ridgerange should be something like y01_y03 or m01_m04
    """
    matbef = ridgerange.split('_')[0]
    mataft = ridgerange.split('_')[1]

    matbef = getsinglemat(matbef)
    mataft = getsinglemat(mataft)

    return(matbef, mataft)


def getbondshocks_process(df, outputoutliers = False):
    """
    In this function I:
    1. Go through each befrate/aftrate combination for ycdi or each rate for yc and a) keep only good rates/maturities b) add a ranking for each observation
    2. Merge together the yield curves from each source keeping the highest ranked observation

    Outputs a spreadsheet with a single variable covering every zone/date shock
    So one source is picked for each zone/date shock
    And then these are aggregated up into a single variable
    """

    # get list of rates to process/rank:{{{
    rate_stems = [col[: -6] for col in df.columns if col.endswith('__rate')]
    befrate_stems = [col[: -9] for col in df.columns if col.endswith('__befrate')]
    aftrate_stems = [col[: -9] for col in df.columns if col.endswith('__aftrate')]

    # verify befrates and aftrates match
    if befrate_stems != aftrate_stems:
        raise ValueError('befrate_stems and aftrate_stems should match.')
    # verify no intersection between rate_stems and befrate_stems
    if len(set(rate_stems) & set(befrate_stems)) > 0:
        raise ValueError("rate_stems and befrate_stems have intersecting components: " + str(set(rate_stems) & set(befrate_stems)) + ".")
    # get list of rates to process/rank:}}}

    # go through rates:{{{
    # create new dataframe to output all sources to (so don't include unneeded variables)
    dfallso = pd.DataFrame(index = df.index)

    for rate_stem in rate_stems:
        rates = list(df[rate_stem + '__rate'])
        mats = list(df[rate_stem + '__mat'])
        ids = list(df[rate_stem + '__id'])
        names = list(df[rate_stem + '__name'])

        # where I store the new rates
        rates_noout = [np.nan] * len(rates)
        mats_noout = [np.nan] * len(rates)
        ids_noout = [np.nan] * len(rates)
        names_noout = [np.nan] * len(rates)
        # where I store a record of the outliers
        rate_outliers = [np.nan] * len(rates)
        mat_outliers = [np.nan] * len(rates)
        id_outliers = [np.nan] * len(rates)
        name_outliers = [np.nan] * len(rates)

        # now go through and replace rates/maturities accordingly
        for i in range(len(rates)):
            
            # general process:{{{
            # stop if nan
            if pd.isnull(rates[i]) is True:
                continue
            if pd.isnull(mats[i]) is True:
                raise ValueError("Weird case where rates[i] exists but mats[i] does not exist. For stem: " + rate_stem + ".")
            
            if len(rates[i]) != len(mats[i]):
                raise ValueError('rates should have the same length as mats. stem: ' + rate_stem + '. rates: ' + str(rates[i]) + '. mats: ' + str(mats[i]) + '.')
            if len(rates[i]) != len(ids[i]):
                raise ValueError('rates should have the same length as ids. stem: ' + rate_stem + '. rates: ' + str(rates[i]) + '. ids: ' + str(ids[i]) + '.')
            if len(rates[i]) != len(names[i]):
                raise ValueError('rates should have the same length as names. stem: ' + rate_stem + '. rates: ' + str(rates[i]) + '. ids: ' + str(names[i]) + '.')

            keepj = []
            observedids = []
            for j in range(len(rates[i])):
                if pd.isnull(rates[i][j]) is True or isinstance(rates[i][j], str) is True or (pd.isnull(ids[i][j]) is False and ids[i][j] in observedids):
                    None
                else:
                    keepj.append(j)
                    # keep record of ids observed for this yield curve and drop next time observe
                    observedids.append(ids[i][j])

            # keep only good rates
            rates[i] = [rates[i][j] for j in keepj]
            mats[i] = [mats[i][j] for j in keepj]
            ids[i] = [ids[i][j] for j in keepj]
            names[i] = [names[i][j] for j in keepj]

            # sort by mats
            matsorder = np.argsort(mats[i])
            rates[i] = [rates[i][j] for j in list(matsorder)]
            mats[i] = [mats[i][j] for j in list(matsorder)]
            ids[i] = [ids[i][j] for j in list(matsorder)]
            names[i] = [names[i][j] for j in list(matsorder)]
            # general process:}}}

            # define no outlier variables
            rates_noout[i] = rates[i]
            mats_noout[i] = mats[i]
            ids_noout[i] = ids[i]
            names_noout[i] = names[i]

            # drop i,j if it is an outlier:{{{
            # need to remove null values beforehand so can compute comparisonchange
            # repeat until no additional outliers found
            oldlen = None
            while len(rates_noout[i]) != oldlen:
                # record what old length was to verify no more changes made
                oldlen = len(rates_noout[i])

                keepj = []
                for j in range(len(rates_noout[i])):

                    if len(rates_noout[i]) < 2:
                        # just set comparison value to be 5 if only one value
                        comparison = 3
                    else:
                        # compute change for interest rates_noout around j
                        # allows me to compare whether or not j is out of the ordinary
                        if j == 0:
                            # if first point compare to change for second point
                            comparison = rates_noout[i][1]
                        elif j == len(rates_noout[i]) - 1:
                            # if last point compare to change for penultimate point
                            comparison = rates_noout[i][j - 1]
                        else:
                            # if middle point compare to change for points before and after
                            comparison = 0.5 * (rates_noout[i][j - 1] + rates_noout[i][j + 1])

                    if abs(rates_noout[i][j] - comparison) > 5:
                        # first check if numbers are very different from nearby numbers
                        # this can sometimes happen if the bond price is mistaken for the bond yield in which case one bond can have a "yield" of 100 rather than 1
                        # also remove if a large change
                        isoutlier = True
                    else:
                        isoutlier = False
                    
                    if isoutlier is False:
                        keepj.append(j)

                # only keep non-outliers
                rates_noout[i] = [rates_noout[i][j] for j in keepj]
                mats_noout[i] = [mats_noout[i][j] for j in keepj]
                ids_noout[i] = [ids_noout[i][j] for j in keepj]
                names_noout[i] = [names_noout[i][j] for j in keepj]

            # get record of outliers
            if pd.isnull(rates_noout[i]) is not True and len(rates_noout[i]) != len(rates[i]):
                rate_outliers[i] = rates[i]
                mat_outliers[i] = mats[i]
                id_outliers[i] = ids[i]
                name_outliers[i] = names[i]

            # drop i,j if it is an outlier:}}}

            # replace with na if []
            if len(mats_noout[i]) == 0:
                rates_noout[i] = np.nan
                mats_noout[i] = np.nan
                ids_noout[i] = np.nan
                names_noout[i] = np.nan

        # update variables
        dfallso[rate_stem + '__rate'] = rates_noout
        dfallso[rate_stem + '__mat'] = mats_noout
        dfallso[rate_stem + '__id'] = ids_noout
        dfallso[rate_stem + '__name'] = names_noout

        # this allows me to see what my outlier rule is excluding
        dfallso[rate_stem + '__rate__outlier'] = rate_outliers
        dfallso[rate_stem + '__mat__outlier'] = mat_outliers
        dfallso[rate_stem + '__id__outlier'] = id_outliers
        dfallso[rate_stem + '__name__outlier'] = name_outliers

    # go through rates:}}}

    # go through befrates/aftrates:{{{
    for befrate_stem in befrate_stems:
        befrates = list(df[befrate_stem + '__befrate'])
        aftrates = list(df[befrate_stem + '__aftrate'])
        mats = list(df[befrate_stem + '__mat'])
        ids = list(df[befrate_stem + '__id'])
        names = list(df[befrate_stem + '__name'])

        # variables I output
        befrates_noout = [np.nan] * len(befrates)
        aftrates_noout = [np.nan] * len(befrates)
        mats_noout = [np.nan] * len(befrates)
        ids_noout = [np.nan] * len(befrates)
        names_noout = [np.nan] * len(befrates)
        # record of outliers to check what I dropped
        befrate_outliers = [np.nan] * len(befrates)
        aftrate_outliers = [np.nan] * len(befrates)
        mat_outliers = [np.nan] * len(befrates)
        id_outliers = [np.nan] * len(befrates)
        name_outliers = [np.nan] * len(befrates)

        # now go through and replace rates/maturities accordingly
        for i in range(len(befrates)):
            
            # general process:{{{
            # continue if nan
            if pd.isnull(befrates[i]) is True or pd.isnull(aftrates[i]) is True:
                continue
            if pd.isnull(mats[i]) is True:
                raise ValueError("befrates and aftrates are not nan but mats is nan. Stem: " + befrate_stem + ".")
            if pd.isnull(ids[i]) is True:
                raise ValueError("befrates and aftrates are not nan but ids is nan. Stem: " + befrate_stem + ".")
            if pd.isnull(names[i]) is True:
                raise ValueError("befrates and aftrates are not nan but names is nan. Stem: " + befrate_stem + ".")

            # verify befrates, aftrates, mats have same length
            if len(befrates[i]) != len(aftrates[i]):
                raise ValueError('befrates should have the same length as aftrates. Stem: ' + befrate_stem + '. befrates: ' + str(befrates[i]) + '. aftrates: ' + str(aftrates[i]) + '.')
            if len(befrates[i]) != len(mats[i]):
                raise ValueError('befrates should have the same length as mats. Stem: ' + befrate_stem + '. befrates: ' + str(befrates[i]) + '. mats: ' + str(mats[i]) + '.')
            if len(befrates[i]) != len(ids[i]):
                raise ValueError('befrates should have the same length as ids. Stem: ' + befrate_stem + '. befrates: ' + str(befrates[i]) + '. ids: ' + str(ids[i]) + '.')
            if len(befrates[i]) != len(names[i]):
                raise ValueError('befrates should have the same length as names. Stem: ' + befrate_stem + '. befrates: ' + str(befrates[i]) + '. names: ' + str(names[i]) + '.')

            # remove elements which are not defined for both befrates and aftrates
            keepj = []
            observedids = []
            for j in range(len(befrates[i])):

                # don't keep if both befrates and aftrates are not defined for a given j
                # also don't keep if they have an absolute difference of more than 1p.p.
                # also don't keep if id observed already
                if pd.isnull(befrates[i][j]) is True or pd.isnull(aftrates[i][j]) is True or isinstance(befrates[i][j], str) is True or isinstance(aftrates[i][j], str) is True or (pd.isnull(ids[i][j]) is False and ids[i][j] in observedids):
                    None
                else:
                    keepj.append(j)
                    # keep record of ids observed for this yield curve and drop next time observe
                    observedids.append(ids[i][j])

                # why drop if same id:
                # getting data from benchmark and non-benchmark bonds so they could point to the same bond with the same id
                # example: RUS 20190322_1030M have the 3 year benchmark bond pointing to RU000A0JSMA2 and direct data from the non-benchmark bond with id RU000A0JSMA2
                # so drop one of these
                # note the rates are not necessarily identical
                # in the aforementioned case they're from =RR and =RRPS so are actually a bit different

            # only keep non-null
            befrates[i] = [befrates[i][j] for j in keepj]
            aftrates[i] = [aftrates[i][j] for j in keepj]
            mats[i] = [mats[i][j] for j in keepj]
            ids[i] = [ids[i][j] for j in keepj]
            names[i] = [names[i][j] for j in keepj]

            # sort by mats
            matsorder = np.argsort(mats[i])
            befrates[i] = [befrates[i][j] for j in list(matsorder)]
            aftrates[i] = [aftrates[i][j] for j in list(matsorder)]
            mats[i] = [mats[i][j] for j in list(matsorder)]
            ids[i] = [ids[i][j] for j in list(matsorder)]
            names[i] = [names[i][j] for j in list(matsorder)]
            # general process:}}}

            # define variable without outlier
            befrates_noout[i] = befrates[i]
            aftrates_noout[i] = aftrates[i]
            mats_noout[i] = mats[i]
            ids_noout[i] = ids[i]
            names_noout[i] = names[i]

            # drop i,j if it is an outlier:{{{
            # drop outliers by comparing change in before and after for different maturities
            # need to remove null values beforehand so can compute comparisonchange
            # repeat until no additional outliers found
            oldlen = None
            while len(befrates_noout[i]) != oldlen:
                # record what old length was to verify no more changes made
                oldlen = len(befrates_noout[i])

                keepj = []
                for j in range(len(befrates_noout[i])):
                    # get the size of the change in this point
                    thischange = aftrates_noout[i][j] - befrates_noout[i][j]

                    if len(befrates_noout[i]) < 2:
                        # can't get a comparison so just keep if:
                        # level satisfies >=-2 and <= 10
                        # changes is not large
                        comparison = 5
                        comparisonchange = 0
                        if befrates_noout[i][j] < -2 or befrates_noout[i][j] > 10 or abs(thischange) > 0.2:
                            isoutlier = True
                        else:
                            isoutlier = False
                    else:
                        # compute change for interest rates around j
                        # allows me to compare whether or not j is out of the ordinary
                        if j == 0:
                            # if first point compare to change for second point
                            comparison = befrates_noout[i][1]
                            comparisonchange = aftrates_noout[i][1] - befrates_noout[i][1]
                        elif j == len(befrates_noout[i]) - 1:
                            # if last point compare to change for penultimate point
                            comparison = befrates_noout[i][j - 1]
                            comparisonchange = aftrates_noout[i][j - 1] - befrates_noout[i][j - 1]
                        else:
                            # if middle point compare to change for points before and after
                            comparison = 0.5 * (befrates_noout[i][j - 1] + befrates_noout[i][j + 1])
                            comparisonchange = 0.5 * (aftrates_noout[i][j - 1] - befrates_noout[i][j - 1] + aftrates_noout[i][j + 1] - befrates_noout[i][j + 1])

                        # now remove outliers by comparing level to comparison level, change to comparison change, and general size of change
                        if abs(befrates_noout[i][j] - comparison) > 5 or abs(aftrates_noout[i][j] - comparison) > 5 or abs(thischange) > 1:
                            # first check if numbers are very different from nearby numbers
                            # this can sometimes happen if the bond price is mistaken for the bond yield in which case one bond can have a "yield" of 100 rather than 1
                            # also remove if a large change
                            isoutlier = True
                        elif abs(thischange) < 0.2:
                            # else if change is small then not an outlier
                            isoutlier = False
                        elif np.sign(thischange) != np.sign(comparisonchange):
                            # if thischange is positive while comparisonchange is negative probably an outlier
                            isoutlier = True
                        elif abs(thischange) < 0.3:
                            # if only a small outlier then only remove is difference from comparisonchange very large
                            if abs(thischange) < 4 * abs(comparisonchange):
                                isoutlier = False
                            else:
                                isoutlier = True
                        elif abs(thischange) < 0.4:
                            if abs(thischange) < 3 * abs(comparisonchange):
                                isoutlier = False
                            else:
                                isoutlier = True
                        else:
                            if abs(thischange) < 2 * abs(comparisonchange):
                                isoutlier = False
                            else:
                                isoutlier = True
                    
                    if isoutlier is False:
                        keepj.append(j)

                # only keep non-outliers
                befrates_noout[i] = [befrates_noout[i][j] for j in keepj]
                aftrates_noout[i] = [aftrates_noout[i][j] for j in keepj]
                mats_noout[i] = [mats_noout[i][j] for j in keepj]
                ids_noout[i] = [ids_noout[i][j] for j in keepj]
                names_noout[i] = [names_noout[i][j] for j in keepj]

            # define outlier variables
            if len(befrates_noout[i]) != len(befrates[i]):
                befrate_outliers[i] = befrates[i]
                aftrate_outliers[i] = aftrates[i]
                mat_outliers[i] = mats[i]
                id_outliers[i] = ids[i]
                name_outliers[i] = names[i]

            # drop i,j if it is an outlier:}}}

            # replace with na if []
            if len(mats_noout[i]) == 0:
                befrates_noout[i] = np.nan
                aftrates_noout[i] = np.nan
                mats_noout[i] = np.nan
                ids_noout[i] = np.nan
                names_noout[i] = np.nan

        dfallso[befrate_stem + '__befrate'] = befrates_noout
        dfallso[befrate_stem + '__aftrate'] = aftrates_noout
        dfallso[befrate_stem + '__mat'] = mats_noout
        dfallso[befrate_stem + '__id'] = ids_noout
        dfallso[befrate_stem + '__name'] = names_noout

        if outputoutliers is True:
            # this allows me to see what my outlier rule is excluding
            dfallso[befrate_stem + '__befrate__outlier'] = befrate_outliers
            dfallso[befrate_stem + '__aftrate_outlier'] = aftrate_outliers
            dfallso[befrate_stem + '__mat__outlier'] = mat_outliers
            dfallso[befrate_stem + '__id__outlier'] = id_outliers
            dfallso[befrate_stem + '__name__outlier'] = name_outliers
            
    # go through befrates/aftrates:}}}

    # get overall ranking:{{{
    for stem in rate_stems + befrate_stems:
        mats = list(dfallso[stem + '__mat'])
        ranks = [np.nan] * len(mats)
        for i in range(len(mats)):

            # stop if maturity is not defined or there are no maturities to consider
            if pd.isnull(mats[i]) is True or len(mats[i]) == 0:
                continue

            # exclude maturities over 15 years when doing source ranking
            mats_max15 = [mat for mat in mats[i] if mat <= 15]

            if len(mats_max15) > 0:
                minmats = min(mats_max15)
                maxmats = max(mats_max15)
                between_3_7 = [mat for mat in mats_max15 if mat > 3 and mat < 7]
            else:
                minmats = None
                maxmats = None
                between_3_7 = []

            # give ranking where 1 is higher than 0 in goodness
            if len(mats_max15) == 0:
                # need to do this case first since when mats == [], minmats and maxmats are undefined
                matsrank = '0'
            elif len(mats_max15) >= 10 and minmats <= 2 and maxmats >= 8 and len(between_3_7) > 0:
                matsrank = '6'
            elif len(mats_max15) >= 8 and minmats <= 2 and maxmats >= 8 and len(between_3_7) > 0:
                matsrank = '5'
            elif len(mats_max15) >= 6 and minmats <= 2 and maxmats >= 8 and len(between_3_7) > 0:
                matsrank = '4'
            elif len(mats_max15) >= 5 and minmats <= 3 and maxmats >= 7:
                matsrank = '3'
            elif len(mats_max15) >= 4 and minmats <= 5 and maxmats >= 5:
                matsrank = '2'
            elif len(mats_max15) >= 4:
                matsrank = '1'
            else:
                matsrank = '0'

            # get second matsrank simply based on the number of maturities available
            # this only matters if the original matsrank and the sourcerank are the same
            if len(mats_max15) > 9999:
                raise ValueError('matsrank2 will not work properly since too many maturities available.')
            matsrank2 = str(len(mats_max15)).zfill(4)

            # get overall ranking
            # note source only matters if matsrank is the same for two sources
            # the underscore isn't really necesary given that these are strings but it looks nicer...
            ranks[i] = matsrank + '_' + matsrank2

        dfallso[stem + '__rank'] = ranks
    # get overall ranking:}}}

    # add in 1so version:{{{
    # data frame for output for single source by shock/zone/date point
    df1so = pd.DataFrame(index = df.index)

    for stem in rate_stems + befrate_stems:
        # get lists of the stem
        if stem + "__rate" in dfallso.columns:
            doingbefrate = False
        else:
            doingbefrate = True

        mats = list(dfallso[stem + "__mat"])
        ranks = list(dfallso[stem + "__rank"])
        if doingbefrate is True:
            befrates = list(dfallso[stem + "__befrate"])
            aftrates = list(dfallso[stem + "__aftrate"])
        else:
            rates = list(dfallso[stem + "__rate"])

        # get version of stem without source
        # i.e. something like ref__ycdi__m1c_1c__rate
        stemnosource = '__'.join(stem.split('__')[1: ])
        source = '__'.join(stem.split('__')[: 1])

        if stemnosource + '__mat' not in df1so:
            # if variable without source defined then just take this as the variable
            mats_new = mats
            ranks_new = ranks
            if doingbefrate is True:
                befrates_new = befrates
                aftrates_new = aftrates
            else:
                rates_new = rates

            sources_new = [np.nan if pd.isnull(befrate) is True else source for befrate in befrates]

        else:
            # if variable without source already defined then need to go through by i and replace if rank is better than existing rank

            # use existing variables in df1so as starting point
            mats_new = list(df1so[stemnosource + "__mat"])
            ranks_new = list(df1so[stemnosource + "__rank"])
            if doingbefrate is True:
                befrates_new = list(df1so[stemnosource + "__befrate"])
                aftrates_new = list(df1so[stemnosource + "__aftrate"])
            else:
                rates_new = list(df1so[stemnosource + "__rate"])
            sources_new = list(df1so[stemnosource + "__source"])

            # then replace new variable if better
            for i in range(len(mats)):
                if pd.isnull(ranks[i]) is False and (pd.isnull(ranks_new[i]) is True or ranks[i] > ranks_new[i]):
                    mats_new[i] = mats[i]
                    ranks_new[i] = ranks[i]
                    if doingbefrate is True:
                        befrates_new[i] = befrates[i]
                        aftrates_new[i] = aftrates[i]
                    else:
                        rates_new[i] = rates[i]
                    sources_new[i] = source

        df1so[stemnosource + "__mat"] = mats_new
        df1so[stemnosource + "__rank"] = ranks_new
        if doingbefrate is True:
            df1so[stemnosource + "__befrate"] = befrates_new
            df1so[stemnosource + "__aftrate"] = aftrates_new
        else:
            df1so[stemnosource + "__rate"] = rates_new
        df1so[stemnosource + "__source"] = sources_new

    # add in 1so version:}}}

    dfallso = dfallso.sort_index(axis = 1)
    df1so = df1so.sort_index(axis = 1)

    return(dfallso, df1so)


def getbondshocks_yc_allso(dfallso, ycnames = None):
    """
    Get yield curves by combining information from all sources
    I use all sources to construct my yield curve measures
    Taking sources by their order in dfallso
    """
    if ycnames is None:
        return(None)


    # verify no bad ycnames
    badycnames = set(ycnames) - {ycname for ycname in ycnames if ycname.startswith('ridgeall__')}
    if len(badycnames) > 0:
        raise ValueError("Bad ycnames: " + str(badycnames) + ".")

    ratestodo = [col for col in dfallso if col.split('__')[-1] in ['rate', 'befrate', 'aftrate']]

    dfyc = pd.DataFrame(index = dfallso.index)

    ridgemats = list(range(1, 31))
    for col in ratestodo:
        # get stem
        stem = '__'.join(col.split('__')[: -1])

        # ref__ycdi__m1c_1c__rate
        colnosource = '__'.join(col.split('__')[1: ])

        # now get rates, mats and ranks
        rates = list(dfallso[col])
        mats = list(dfallso[stem + '__mat'])
        ranks = list(dfallso[stem + '__rank'])

        # ridgealls:{{{
        ridgealls = [ycname for ycname in ycnames if ycname.startswith('ridgeall__')]
        for ridgename in ridgealls:
            window = ridgename.replace('ridgeall__', '', 1)
            mat1 = window.split('_')[0]
            mat2 = window.split('_')[1]

            if colnosource + '__' + ridgename not in dfyc:
                ridges = [np.nan for i in range(len(rates))]
            else:
                ridges = dfyc[colnosource + '__' + ridgename]

            for i in range(len(rates)):
                if pd.isnull(rates[i]) is True:
                    continue
                
                # only add if does not already exist
                if np.isnan(ridges[i]):
                    ridges[i] = ridge_yieldcurve(mats[i], rates[i], mat1, mat2, matout_string = True)

            # add into dataset
            dfyc[colnosource + '__' + ridgename] = ridges
        # ridgealls:}}}

    # replace [na, na, ...] with na:{{{
    # long lists of nas can take up a lot of file space so replace them with a single na
    ridgecols = [col for col in dfyc if col.split('__')[-1].startswith('ridgeall')]
    for ridgecol in ridgecols:
        ridges = list(dfyc[ridgecol])
        for i in range(len(ridges)):
            # if already na continue
            if pd.isnull(ridges[i]) is True:
                continue
            # if ridges[i] is only made up of nas
            if len(set(ridges[i]) - set([np.nan])) == 0:
                ridges[i] = np.nan
        dfyc[ridgecol] = ridges
    # replace [na, na, ...] with na:}}}

    # add di:{{{
    # only do this for befrates/aftrates not rates
    ratestodo = [col for col in dfallso if col.split('__')[-1] in ['befrate']]
    for col in ratestodo:
        colnosource = '__'.join(col.split('__')[1: ])
        for ycname in ycnames:
            colname = colnosource + '__' + ycname
            dfyc[colname.replace('__befrate__', '__di__')] = dfyc[colname.replace('__befrate__', '__aftrate__')] - dfyc[colname]

    # add di:}}}

    dfyc = dfyc.sort_index(axis = 1)

    return(dfyc)


def getbondshocks_yc_1so(df1so, ycnames = None, printdetails = False):
    """
    Computes yield curve on the best source for each rate using the best source in df1so
    Note this does not compute my main ridge yield curve measure which uses all potential sources

    ycnames is a list including up to ['ols', 'nslu', 'ridge1_1so', 'ridge2_1so', 'nsme']
    The function is only run on the specified elements in ycnames
    They are saved separately

    ols: OLS
    nslu: Nelson-Siegel with Luphord code
    nsme: Nelson-Siegel with my code (slow so I leave it out by default)
    ridge1_1so/ridge2_1so: nonparametric method where I'm basically taking the closest yield curve (but not my main version of this - since I'm not including all sources)

    Note nsme takes a long time to run (as in 100x the others since I'm running many regressions probably inefficiently during it). The others are fairly quick.
    """

    if ycnames is None:
        return(None)

    # verify no bad ycnames
    badycnames = set(ycnames) - {ycname for ycname in ycnames if ycname.startswith('ridge__')} - {ycname for ycname in ycnames if ycname.startswith('nslu_')}
    if len(badycnames) > 0:
        raise ValueError("Bad ycnames: " + str(badycnames) + ".")

    ratestodo = [col for col in df1so if col.split('__')[-1] in ['rate', 'befrate', 'aftrate']]

    dfyc = pd.DataFrame(index = df1so.index)

    # create rates/befrates/aftrates:{{{
    mats_output = list(range(1, 31))
    mats_output_max15 = list(range(1, 16))
    for col in ratestodo:
        if printdetails is True:
            print(str(datetime.datetime.now()) + ' Starting column: ' + col + '.')
        # get stem
        stem = '__'.join(col.split('__')[: -1])

        # now get rates, mats and ranks
        rates = list(df1so[col])
        mats = list(df1so[stem + '__mat'])
        ranks = list(df1so[stem + '__rank'])

        # get versions with >15 year maturities removed
        mats_max15 = [np.nan] * len(rates)
        rates_max15 = [np.nan] * len(rates)
        for i in range(len(rates)):
            if pd.isnull(mats[i]) is True:
                continue
            mats_max15_elements = [j for j in range(len(mats[i])) if mats[i][j] <= 15]
            mats_max15[i] = [mats[i][j] for j in mats_max15_elements]
            rates_max15[i] = [rates[i][j] for j in mats_max15_elements]
            
        # ridges:{{{
        ridges = [ycname for ycname in ycnames if ycname.startswith('ridge__')]
        for ridgename in ridges:
            window = ridgename.replace('ridge__', '', 1)
            mat1 = window.split('_')[0]
            mat2 = window.split('_')[1]
            
            rateoutputs = [np.nan] * len(rates)

            for i in range(len(rates)):
                # stop if no rates to consider
                if pd.isnull(rates[i]) is True:
                    continue
                
                rateoutputs[i] = ridge_yieldcurve(mats[i], rates[i], mat1, mat2, matout_string = True)

            # add into dataset
            dfyc[col + '__' + ridgename] = rateoutputs
        # ridges:}}}

        # nslu:{{{
        nslus = [ycname for ycname in ycnames if ycname.startswith('nslu_')]

        # general nslu computation:{{{
        if len(nslus) > 0:

            params = [np.nan] * len(rates)
            curves = [np.nan] * len(rates)

            for i in range(len(rates)):
                if pd.isnull(rates[i]) is True or ranks[i] < '1':
                    continue

                success = False
                try:
                    # note dividing by 100
                    curve, status = calibrate_ns_ols(np.array(mats_max15[i]), np.array(rates_max15[i]) / 100, tau0 = 1.0)
                    if status['success'] is True:
                        success = True
                except Exception:
                    None

                if success is True:
                    paramdict = vars(curve)

                    # drop bad parameter:{{{
                    """
                    A few cases where algorithm does not work properly
                    for example Turkey 20181025d with following parameters:
                    rates: [24.026, 25.63, 24.4, 21.61, 18.28]
                    maturities: [0.5, 0.898, 1.799, 2.91, 9.369]
                    NSLU parameters: [0.24807759929782192, -0.0001242337575754526, -5.045433797156877e-06, -0.3656479693246273]
                    NSLU output: [24.747427383791656, 24.36880120830213, 20.891315860711074, -13.358179418230915, -358.02256000366066, -3765.5866839269547, -35175.06951371506, -274212.8363491629, -1012341.2905774283, 24006077.95079504, 869145377.1650635, 19802051314.59813, 388680603189.31506, 7092446873687.183, 124017502376382.94]
                    issue may be that all but one rate has a maturity of <3 years
                    last NSLU parameter is negative which I think is driving strange numbers so drop if that happens
                    """
                    if paramdict['tau'] <= 0:
                        params[i] = 'NSFailed'
                        continue
                    # drop bad parameter:}}}

                    params[i] = [paramdict['beta0'], paramdict['beta1'], paramdict['beta2'], paramdict['tau']]
                    curves[i] = curve

                else:
                    if False:
                        print('\nFailed Nelson-Siegel:')
                        print('Col: ' + str(col) + '.')
                        print('Index: ' + str(i) + '.')
                        print('Maturities: ' + str(mats_max15[i]) + '.')
                        print('Yields: ' + str(rates_max15[i]) + '.')

                    params[i] = 'NSFailed'

            # add into dataset
            dfyc[col + '__nslu__params'] = params
        # general nslu computation:}}}

        # go through each desired maturity/rank
        # specify something like nslu__y4 or nslu_r5__m06
        for ycnamenslu in nslus:
            firstpart = ycnamenslu.split('__')[0]
            if firstpart == 'nslu':
                rankmin = 4
            else:
                rankmin = int(firstpart.replace('nslu_r', ''))
            secondpart = ycnamenslu.split('__')[1]
            sptime = secondpart[0]
            maturity = int(secondpart[1: ])
            if sptime == 'y' or sptime == 'z':
                None
            elif sptime == 'm':
                maturity = maturity / 12
            elif sptime == 'd':
                maturity = maturity / 365
            else:
                raise ValueError('ycnamenslu misspecified: ' + ycnamenslu + '.')
                
            rateoutputs = [np.nan] * len(rates)
            for i in range(len(rates)):
                if pd.isnull(curves[i]):
                    continue
                if ranks[i] < str(rankmin):
                    continue

                # check to ensure not defining bad variables:{{{
                # Issue with Nelson-Siegel estimation:
                # - If I only have data for year x onwards, the data pre-year x can be very off
                # - For example, when I had data for years 2-15 which went from 16-20 (with a bit of a sudden jump), Nelson-Siegel estimated the yield for a 1-year bond to be 14000%
                # - So should using the estimates for Nelson-Siegel for years for which I do not have the same or a lower maturity bond
                # - Less of an issue for maturities that exceed my higest maturity because of the way the exponentials in Nelson-Siegel work
                # - When computing change in yield, drop very high interest rates and large changes as further check
                # - Note need to adjust which early maturities drop depending on strictness of shock i.e. with s1 can only compute 5+ year change in yield compared to with s5 can compute 1+ year change

                # don't worry about doing this for ycdi_for where I've already checked the appropriate bonds to include
                earliestmat = mats_max15[i][0]
                if earliestmat > maturity:
                    continue

                # check to ensure not defining bad variables:}}}

                # note multiplying by 100 to return to percentage form
                rateoutputs[i] = curves[i](maturity) * 100

            # add into dataset
            dfyc[col + '__' + ycnamenslu] = rateoutputs
        # nslu:}}}

    # create rates/befrates/aftrates:}}}

    # add di:{{{
    # only do this for befrates/aftrates not rates
    ratestodo = [col for col in df1so if col.split('__')[-1] in ['befrate']]
    for col in ratestodo:

        for ycname in ycnames:
            colname = col + '__' + ycname
            dfyc[colname.replace('__befrate__', '__di__')] = dfyc[colname.replace('__befrate__', '__aftrate__')] - dfyc[colname]

    # add di:}}}

    return(dfyc)


def getindividualbonds(df):
    """
    Convert and unprocessed or processed (with all sources) data frame back to individual bonds
    Rather than lists
    """

    # get list of rates to process/rank:{{{
    rate_stems = [col[: -6] for col in df.columns if col.endswith('__rate')]
    befrate_stems = [col[: -9] for col in df.columns if col.endswith('__befrate')]
    aftrate_stems = [col[: -9] for col in df.columns if col.endswith('__aftrate')]

    # verify befrates and aftrates match
    if befrate_stems != aftrate_stems:
        raise ValueError('befrate_stems and aftrate_stems should match.')
    # verify no intersection between rate_stems and befrate_stems
    if len(set(rate_stems) & set(befrate_stems)) > 0:
        raise ValueError("rate_stems and befrate_stems have intersecting components: " + str(set(rate_stems) & set(befrate_stems)) + ".")
    # get list of rates to process/rank:}}}

    # go through rates:{{{
    stemstoconcat = []
    for rate_stem in rate_stems:
        rates = list(df[rate_stem + '__rate'])
        mats = list(df[rate_stem + '__mat'])
        ids = list(df[rate_stem + '__id'])
        names = list(df[rate_stem + '__name'])

        rowstoconcat = []

        # now go through and replace rates/maturities accordingly
        for i in range(len(rates)):
            outdict = {}
            
            # stop if nan
            if isinstance(rates[i], list) is False and pd.isnull(rates[i]) is True:
                # add empty row with no columns
                rowstoconcat.append(pd.DataFrame(index = [0]))
                continue

            if isinstance(rates[i], list) is True and len(rates[i]) == 0:
                # add empty row with no columns
                rowstoconcat.append(pd.DataFrame(index = [0]))
                continue

            for j in range(len(rates[i])):
            
                outdict[rate_stem + '__' + names[i][j] + '__rate'] = [rates[i][j]]
                outdict[rate_stem + '__' + names[i][j] + '__mat'] = [mats[i][j]]
                outdict[rate_stem + '__' + names[i][j] + '__id'] = [ids[i][j]]

            dfrow = pd.DataFrame(outdict)
            rowstoconcat.append(dfrow)

        dfstem = pd.concat(rowstoconcat)
        dfstem.index = df.index

        stemstoconcat.append(dfstem)

    if len(stemstoconcat) > 0:
        df2 = pd.concat(stemstoconcat, axis = 1)
    # go through rates:}}}

    # go through befrates/aftrates:{{{
    stemstoconcat = []
    for befrate_stem in befrate_stems:
        befrates = list(df[befrate_stem + '__befrate'])
        aftrates = list(df[befrate_stem + '__aftrate'])
        mats = list(df[befrate_stem + '__mat'])
        ids = list(df[befrate_stem + '__id'])
        names = list(df[befrate_stem + '__name'])

        rowstoconcat = []

        # now go through and replace rates/maturities accordingly
        for i in range(len(befrates)):
            outdict = {}
            
            # stop if nan
            if isinstance(befrates[i], list) is False and pd.isnull(befrates[i]) is True:
                # add empty row with no columns
                rowstoconcat.append(pd.DataFrame(index = [0]))
                continue

            if isinstance(befrates[i], list) is True and len(befrates[i]) == 0:
                # add empty row with no columns
                rowstoconcat.append(pd.DataFrame(index = [0]))
                continue

            for j in range(len(befrates[i])):
            
                outdict[befrate_stem + '__' + names[i][j] + '__befrate'] = [befrates[i][j]]
                outdict[befrate_stem + '__' + names[i][j] + '__aftrate'] = [aftrates[i][j]]
                outdict[befrate_stem + '__' + names[i][j] + '__mat'] = [mats[i][j]]
                outdict[befrate_stem + '__' + names[i][j] + '__id'] = [ids[i][j]]

            dfrow = pd.DataFrame(outdict)
            rowstoconcat.append(dfrow)

        dfstem = pd.concat(rowstoconcat)
        dfstem.index = df.index

        stemstoconcat.append(dfstem)

    if len(stemstoconcat) > 0:
        df2 = pd.concat(stemstoconcat, axis = 1)
    # go through befrates/aftrates:}}}

    df2 = df2.sort_index(axis = 1)

    return(df2)

    
def forwarddiff(df, prefix, ycnamestart, ycnameend):
    """
    Structure of difference variable
    ycdi__m1h_1h__fdi__YCNAME1__YCNAME2
    prefix = 'ycdi__m1h_1h'
    ycnamestart = 'ridge__y01_y03'
    ycnameend = 'nslu__y05'
    """

    # get maturity of start:{{{
    if ycnamestart.startswith('ridge'):
        matlowpart = ycnamestart.split('__')[1].split('_')[0]
        mathighpart = ycnamestart.split('__')[1].split('_')[1]

        matlowletter = matlowpart[0]
        mathighletter = mathighpart[0]

        if matlowpart[1: ].isnumeric() is False:
            raise ValueError("matlowpart fails as latter part not numeric: " + str(ycnamestart) + ".")
        matlow = int(matlowpart[1: ])

        if mathighpart[1: ].isnumeric() is False:
            raise ValueError("mathighpart fails as latter part not numeric: " + str(ycnamestart) + ".")
        mathigh = int(mathighpart[1: ])

        if matlowletter in ['y', 'z']:
            None
        elif matlowletter == 'm':
            matlow = matlow / 12
        elif matlowletter == 'd':
            matlow = matlow / 365
        else:
            raise ValueError('Wrong format of matlowletter: ' + ycnamestart + '.')

        if mathighletter in ['y', 'z']:
            None
        elif mathighletter == 'm':
            mathigh = mathigh / 12
        elif mathighletter == 'd':
            mathigh = mathigh / 365
        else:
            raise ValueError('Wrong format of mathighletter: ' + ycnamestart + '.')

        matstart = (matlow + mathigh) / 2

    elif ycnamestart.startswith('nslu'):

        matpart = ycnamestart.split('__')[1]
        matstartletter = matpart[0]
        matstart = int(matpart[1: ])

        if matstartletter in ['y', 'z']:
            None
        elif matstartletter == 'm':
            matstart = matstart / 12
        elif matstartletter == 'd':
            matstart = matstart / 365
        else:
            raise ValueError('Wrong format of matstartletter: ' + ycnamestart + '.')
    else:
        raise ValueError('ycnamestart misspecified: ' + ycnamestart + '.')
    # get maturity of start:}}}

    # get maturity of end:{{{
    if ycnameend.startswith('ridge'):
        matlowpart = ycnameend.split('__')[1].split('_')[0]
        mathighpart = ycnameend.split('__')[1].split('_')[1]

        matlowletter = matlowpart[0]
        mathighletter = mathighpart[0]

        if matlowpart[1: ].isnumeric() is False:
            raise ValueError("matlowpart fails as latter part not numeric: " + str(ycnameend) + ".")
        matlow = int(matlowpart[1: ])

        if mathighpart[1: ].isnumeric() is False:
            raise ValueError("mathighpart fails as latter part not numeric: " + str(ycnameend) + ".")
        mathigh = int(mathighpart[1: ])

        if matlowletter in ['y', 'z']:
            None
        elif matlowletter == 'm':
            matlow = matlow / 12
        elif matlowletter == 'd':
            matlow = matlow / 365
        else:
            raise ValueError('Wrong format of matlowletter: ' + ycnameend + '.')

        if mathighletter in ['y', 'z']:
            None
        elif mathighletter == 'm':
            mathigh = mathigh / 12
        elif mathighletter == 'd':
            mathigh = mathigh / 365
        else:
            raise ValueError('Wrong format of mathighletter: ' + ycnameend + '.')

        matend = (matlow + mathigh) / 2

    elif ycnameend.startswith('nslu'):

        matpart = ycnameend.split('__')[1]
        matendletter = matpart[0]
        matend = int(matpart[1: ])

        if matendletter in ['y', 'z']:
            None
        elif matendletter == 'm':
            matend = matend / 12
        elif matendletter == 'd':
            matend = matend / 365
        else:
            raise ValueError('Wrong format of matendletter: ' + ycnameend + '.')
    else:
        raise ValueError('ycnameend misspecified: ' + ycnameend + '.')
    # get maturity of end:}}}

    df[prefix + '__fdi__' + ycnamestart + '__' + ycnameend] = (matend * df[prefix + '__di__' + ycnameend] - matstart * df[prefix + '__di__' + ycnamestart]) / (matend - matstart)
    
    return(df)


def forwarddiff_varname(df, varnames):
    if isinstance(varnames, str):
        varnames = [varnames]

    for varname in varnames:
        prefix = '__'.join(varname.split('__')[0: 2])
        ycnamestart = '__'.join(varname.split('__')[3: 5])
        ycnameend = '__'.join(varname.split('__')[5: 7])
        df = forwarddiff(df, prefix, ycnamestart, ycnameend)

    return(df)


def forwarddiff_test():
    df = pd.DataFrame({'ycdi__m1h_1h__di__ridge__y00_y02': [0.1, 0.2], 'ycdi__m1h_1h__di__ridge__y01_y03': [0.2, 0.3]})
    df = forwarddiff(df, 'ycdi__m1h_1h', 'ridge__y00_y02', 'ridge__y01_y03')
    """
    Should find 0.3 for first row
    (0.2 * 2 - 0.1 * 1) / (2 - 1) = 0.3
    """
    print(df)

    # same but with the _varname version of the function
    df = pd.DataFrame({'ycdi__m1h_1h__di__ridge__y00_y02': [0.1, 0.2], 'ycdi__m1h_1h__di__ridge__y01_y03': [0.2, 0.3]})
    df = forwarddiff_varname(df, 'ycdi__m1h_1h__fdi__ridge__y00_y02__ridge__y01_y03')
    """
    Should find 0.3 for first row
    (0.2 * 2 - 0.1 * 1) / (2 - 1) = 0.3
    """
    print(df)


def graphycfit(df, timeframe, zone, time, timevar, nsluyears = None, plotshow = False, savename = None, ridgerange = None, includeafter = True, xlim = None, ylim = None):
    """
    Input df containing:
    - ycdi__m1h_1h__befrate
    - ycdi__m1h_1h__aftrate
    - ycdi__m1h_1h__mat
    - ycdi__m1h_1h__befrate__nslu__y02 etc.
    - ycdi__m1h_1h__aftrate__nslu__y02 etc.
    - zone
    - timevariable

    If ridgerange is None then plot a nslu graph using nsluyears
    Otherwise plot a ridge graph. ridgerange = 'y01_y03' etc.

    """

    if ridgerange is None:
        if nsluyears is None:
            nsluyears = ['y01', 'y02', 'y03', 'y04', 'y05', 'y06', 'y07', 'y08', 'y09', 'y10', 'y11', 'y12', 'y13', 'y14', 'y15']
        nslu = True
    else:
        if nsluyears is not None:
            raise ValueError("nsluyears should not be defined if ridgename is not None.")
        nslu = False
    
    # get names of variables I'm interested in
    name_befrate = 'ycdi__' + timeframe + '__befrate'
    name_aftrate = 'ycdi__' + timeframe + '__aftrate'
    name_mat = 'ycdi__' + timeframe + '__mat'
    if nslu is True:
        name_befrate_fitteds = ['ycdi__' + timeframe + '__befrate__nslu__' + year for year in nsluyears]
        name_aftrate_fitteds = ['ycdi__' + timeframe + '__aftrate__nslu__' + year for year in nsluyears]
    else:
        name_befrate_fitteds = ['ycdi__' + timeframe + '__befrate__ridge__' + ridgerange]
        name_aftrate_fitteds = ['ycdi__' + timeframe + '__aftrate__ridge__' + ridgerange]

    # restrict only to relevant time
    df2 = df[df[timevar] == time]
    if len(df2) != 1:
        print(df2)
        raise ValueError('Not a single time when timevar equals ' + time + '.')

    # get variables for relevant times as lists
    befrates = df2[name_befrate].iloc[0]
    aftrates = df2[name_aftrate].iloc[0]
    mats = df2[name_mat].iloc[0]
    befrate_fitteds = df2[name_befrate_fitteds].values.tolist()[0]
    aftrate_fitteds = df2[name_aftrate_fitteds].values.tolist()[0]

    # get maturities for fitteds
    if nslu is True:
        mats_fitteds = [year for year in nsluyears]
        for i in range(len(mats_fitteds)):
            mats_fitteds[i] = getsinglemat(mats_fitteds[i])
    else:
        matbef, mataft = getridgemats(ridgerange)
        mats_fitteds = [(matbef + mataft) / 2]

    # restrict befrates/aftrates to only show within the ridgerange
    if nslu is True:
        matbef = 0
        mataft = 15
    else:
        matbef, mataft = getridgemats(ridgerange)
    mats_istar = [i for i in range(len(mats)) if mats[i] >= matbef and mats[i] <= mataft]
    befrates = [befrates[i] for i in mats_istar]
    aftrates = [aftrates[i] for i in mats_istar]
    mats = [mats[i] for i in mats_istar]

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Interest Rate (%)')

    if nslu is True:
        fittedstyle = '-'
        markersize = 5
    else:
        fittedstyle = '_'
        markersize = 15
    plt.plot(mats, befrates, 'bo', label = 'Before Underlying')
    plt.plot(mats_fitteds, befrate_fitteds, 'b' + fittedstyle, markersize = markersize, label = 'Before Fitted')
    if includeafter is True:
        plt.plot(mats, aftrates, 'ro', label = 'After Underlying')
        plt.plot(mats_fitteds, aftrate_fitteds, 'r' + fittedstyle, markersize = markersize, label = 'After Fitted')

    plt.legend()

    if savename is not None:
        plt.savefig(savename)

    if plotshow is True:
        plt.show()

    plt.clf()


# Long to Wide Interest Rates for Many Zones/Same Dates:{{{1
def getdomestic_fromlong(dflong, dfdates, datevar):
    """
    dfdates gives a list of the dates associated with each zone, so it contains a 'zone' and a datevar variable
    We then restrict dflong to only the dates associated with each zone
    """

    dfdomestic = dflong.merge(dfdates[['zone', datevar]], on = ['zone', datevar], how = 'right')

    return(dfdomestic)


def getforeign_fromlong(dflong, dfdates, datevar, zonesinclude = None, suffix = 'for'):
    """
    This computes the global mean excluding the local zone for variables around the mean
    All the variables in dfdates except 'zone' and datevar must be floats
    
    """
    if zonesinclude is not None:
        dflong = dflong[dflong['zone'].isin(zonesinclude)].copy()

    # reshape long to wide
    dfwide = pd.pivot(dflong, index = datevar, columns = 'zone', values = [col for col in dflong.columns if col not in ['zone', datevar]])

    varnames = sorted(list(set([col[0] for col in dfwide.columns])))
    zones = sorted(list(set([col[1] for col in dfwide.columns])))

    # adjust before merge
    # rename as single word so don't have issues when merging with dfdates
    dfwide.columns = [col[0] + '__' + col[1] for col in dfwide.columns]
    dfwide = dfwide.reset_index()

    # merge in zone so now have zone/date pair and all interest rates of all zones
    dfwide = dfwide.merge(dfdates, on = datevar, how = 'right')

    # for each zone in the zones in the dataset, set that dfwide data to nan
    for zone in zones:
        dfwide.loc[dfwide['zone'] == zone, [col for col in dfwide.columns if col.split('__')[-1] == zone]] = np.nan

    # FOR EACH ZONE
    dfglobal = pd.DataFrame({'zone': dfwide['zone'], datevar: dfwide[datevar]})
    for varname in varnames:
        # get all columns relating to a given varname
        dfwide2 = dfwide[[col for col in dfwide.columns if '__'.join(col.split('__')[: -1]) == varname]]
        # define a single global varname for that variable
        dfglobal[varname.split('__')[0] + '_' + suffix + '__' + '__'.join(varname.split('__')[1: ])] = dfwide2.mean(axis = 1)

    return(dfglobal)


def getzones_fromlong(dflong, datevar, dfdates = None, zonesinclude = None):
    """
    df should be a dataset where I have zones covering the same dates/times where dates/times are given by datevar

    This function then reshapes the data to get the interest rate data for each zone in zones for the dates

    If dfdates is specified then match the dates to the relevant zones in which those dates occurred
    zonesinclude: Only get interest rates for zones in zonesinclude if specified
    """
    if zonesinclude is not None:
        dflong = dflong[dflong['zone'].isin(zonesinclude)].copy()

    # reshape to long
    dfwide = pd.pivot(dflong, index = datevar, columns = 'zone', values = [col for col in dflong.columns if col not in['zone', datevar]])

    dfwide.columns = [col[0].split('__')[0] + '_zone_' + col[1].lower() + '__' + '__'.join(col[0].split('__')[1: ]) for col in dfwide.columns]
    dfwide = dfwide.reset_index()

    # if there are object dtypes in pandas pivot seems to make all variables object afterwards
    # therefore convert back to numeric where possible
    # dfwide = dfwide.astype(float, errors = 'ignore')
    for var in dfwide.columns:
        dfwide[var] = pd.to_numeric(dfwide[var], errors = 'ignore')

    if dfdates is not None:
        dfwide = dfwide.merge(dfdates[['zone', datevar]], on = datevar, how = 'right')

    return(dfwide)


