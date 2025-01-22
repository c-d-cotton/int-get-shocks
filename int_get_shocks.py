#!/usr/bin/env python3
"""
"""


import os
from pathlib import Path
import sys
from warnings import simplefilter

try:
    __projectdir__ = Path(os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/'))
except NameError:
    __projectdir__ = Path(os.path.abspath(""))


import copy
import datetime
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
from nelson_siegel_svensson.calibrate import NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols
from nelson_siegel_svensson.calibrate import NelsonSiegelSvenssonCurve
import numpy as np
import pandas as pd
import pickle

from event_shock_func import *

# turn off annoying warning (DataFrame is highly fragmented...)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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
reldict_daily_extended['m90c_b7'] = {'rday': -90, 'open': False, 'weekendignore': True, 'bfill': 7}

# Shockdict Daily:{{{2
shockdict_daily_basic = {}
shockdict_daily_basic["1d"] = ["m1c", "0c"]

shockdict_daily_extended = {}
shockdict_daily_extended["2d"] = ["m1c", "1c"]
shockdict_daily_extended["m1c"] = ["m1c_f7", "m1c_f7"]
shockdict_daily_extended["m90m1c"] = ["m90c_b7", "m1c_f7"]
shockdict_daily_extended["2do"] = ["m1c", "1o"]

# Reldict Return Daily:{{{2
reldictreturn_daily_extended = {'m1c': 'm1c_f7'}

# Reldict Intraday:{{{2
# reldict to apply as input in function to getdfintrarel_reldict
reldict_intra_basic = {}
# if event is 1600M, this would take the first trade at 1550M-1559M or if that's not defined then the last trade at 1500M-1549M (with 5 minute interval data)
reldict_intra_basic['m1h'] = {'rpos': -2, 'open': True, 'bfill': 2, 'ffill': 10, 'ffillfirst': False}
# if event is 1600M, this would take the last trade at 1610M-1619M or if that's not defined then the first trade at 1620M-1659M (with 5 minute interval data)
reldict_intra_basic['1h'] = {'rpos': 3, 'open': False, 'ffill': 2, 'bfill': 8, 'ffillfirst': True}
# if event is 1600M, this would take the first trade at 1550M-1559M (with 5 minute interval data)
reldict_intra_basic['m10m'] = {'rpos': -2, 'open': True, 'bfill': 2}
# if event is 1600M, this would take the last trade at 1610M-1619M (with 5 minute interval data)
reldict_intra_basic['20m'] = {'rpos': 3, 'open': False, 'ffill': 2}
# and define precise 15 minute intervals
# if event is 1600M, this would take the last trade at 1500M-1549M (with 5 minute interval data)
reldict_intra_basic['m1h'] = {'rpos': -3, 'open': False, 'ffill': 9}
# if event is 1600M, this would take the first trade at 1610M-1659M (with 5 minute interval data)
reldict_intra_basic['1h'] = {'rpos': 2, 'open': True, 'bfill': 9}

reldict_intra_extended = copy.deepcopy(reldict_intra_basic)
# if event is 1600M, this would take the first trade at 1550M-1559M or if that's not defined then the last trade at 1530M-1549M (with 5 minute interval data)
reldict_intra_extended['m30m'] = {'rpos': -2, 'open': True, 'bfill': 2, 'ffill': 4, 'ffillfirst': False}
# if event is 1600M, this would take the last trade at 1610M-1619M or if that's not defined then the first trade at 1620M-1630M (with 5 minute interval data)
reldict_intra_extended['30m'] = {'rpos': 3, 'open': False, 'ffill': 2, 'bfill': 2, 'ffillfirst': True}
# take point 12 hours from now i.e. 12 * 12 - 1 (since 12 5 minute intervals in each hour)
# reldict_intra_extended['12h'] = {'rpos': 143, 'open': False, 'ffill': 141}
# take point 24 hours from now i.e. 24 * 12 - 1
reldict_intra_extended['24h'] = {'rpos': 287, 'open': False, 'ffill': 285}

# Shockdict Intraday:{{{2
# shockdict to apply as input in function to getbefaft
shockdict_intra_basic = {}
shockdict_intra_basic["2h"] = ["m1h", "1h"]
shockdict_intra_basic["30m"] = ["m10m", "20m"]

shockdict_intra_extended = {}
shockdict_intra_extended["1h"] = ["m30m", "30m"]
# shockdict_intra_extended["12h"] = ["m2h", "12h"]
shockdict_intra_extended["24h"] = ["m1h", "24h"]
# shockdict_intra_extended["48h"] = ["m2h", "48h"]
# shockdict_intra_extended["72h"] = ["m2h", "72h"]
# shockdict_intra_extended["96h"] = ["m2h", "96h"]
shockdict_intra_extended["168h"] = ["m1h", "168h"]

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
def mergezonedir(mergedir, picklefiles = False, includeonlyprefix = None):
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

        if includeonlyprefix is not None:
            dfzone = dfzone[[col for col in dfzone.columns if col.startswith(includeonlyprefix)]]

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


def getranking(mats):
    """
    Get ranking of a list of maturities for bonds
    """

    # decided to use all mats rather than just mats_max15
    # exclude maturities over 15 years when doing source ranking
    # mats_max15 = [mat for mat in mats if mat <= 15]

    if len(mats) > 0:
        minmats = min(mats)
        maxmats = max(mats)
        between_3_7 = [mat for mat in mats if mat > 3 and mat < 7]
    else:
        minmats = None
        maxmats = None
        between_3_7 = []

    # give ranking where 1 is higher than 0 in goodness
    if len(mats) == 0:
        # need to do this case first since when mats == [], minmats and maxmats are undefined
        matsrank = '0'
    elif len(mats) >= 10 and minmats <= 2 and maxmats >= 8 and len(between_3_7) > 0:
        matsrank = '6'
    elif len(mats) >= 8 and minmats <= 2 and maxmats >= 8 and len(between_3_7) > 0:
        matsrank = '5'
    elif len(mats) >= 6 and minmats <= 2 and maxmats >= 8 and len(between_3_7) > 0:
        matsrank = '4'
    elif len(mats) >= 5 and minmats <= 3 and maxmats >= 7:
        matsrank = '3'
    elif len(mats) >= 4 and minmats <= 5 and maxmats >= 5:
        matsrank = '2'
    elif len(mats) >= 4:
        matsrank = '1'
    else:
        matsrank = '0'

    # get second matsrank simply based on the number of maturities available
    # this only matters if the original matsrank and the sourcerank are the same
    if len(mats) > 9999:
        raise ValueError('matsrank2 will not work properly since too many maturities available.')
    matsrank2 = str(len(mats)).zfill(4)

    overallrank = float(matsrank + '.' + matsrank2)

    return(overallrank)


def getridgemats(ridgerange):
    """
    ridgerange should be something like y01_y03 or m01_m04
    """
    matbef = ridgerange.split('_')[0]
    mataft = ridgerange.split('_')[1]

    matbef = getsinglemat(matbef)
    mataft = getsinglemat(mataft)

    return(matbef, mataft)


def getbondshocks_process(df):
    """
    In this function I:
    1. Go through each befrate/aftrate combination for ycdi or each rate for yc and a) keep only good rates/maturities b) add a ranking for each observation
    2. Merge together the yield curves from each source keeping the highest ranked observation

    Outputs a spreadsheet with a single variable covering every zone/date shock
    So one source is picked for each zone/date shock
    And then these are aggregated up into a single variable
    """

    dfout = pd.DataFrame(index = df.index)

    # merge columns together to get iout:{{{
    prefixes = sorted(list(set(['__'.join(col.split('__')[: -1]) for col in df.columns])))
    for prefix in prefixes:
        # prefix is something like z_deu_ref__ycdi__m1c_1c
        befrates = list(df[prefix + '__befrate'])
        aftrates = list(df[prefix + '__aftrate'])
        ids = list(df[prefix + '__id'])
        mats = list(df[prefix + '__mat'])
        names = list(df[prefix + '__name'])

        # break up prefix into two parts
        # z_deu_ref
        prefix_source = prefix.split('__')[0]
        # ycdi__m1c_1c
        prefix_nonsource = '__'.join(prefix.split('__')[1: ])

        if 'iout__' + prefix_nonsource + '__befrate' in dfout.columns:
            # need to add this source to existing lists
            befrates_out = list(dfout['iout__' + prefix_nonsource + '__befrate'])
            aftrates_out = list(dfout['iout__' + prefix_nonsource + '__aftrate'])
            ids_out = list(dfout['iout__' + prefix_nonsource + '__id'])
            mats_out = list(dfout['iout__' + prefix_nonsource + '__mat'])
            names_out = list(dfout['iout__' + prefix_nonsource + '__name'])
            sources_out = list(dfout['iout__' + prefix_nonsource + '__source'])
        else:
            # create new lists
            befrates_out = [[] for i in range(len(dfout))]
            aftrates_out = [[] for i in range(len(dfout))]
            ids_out = [[] for i in range(len(dfout))]
            mats_out = [[] for i in range(len(dfout))]
            names_out = [[] for i in range(len(dfout))]
            sources_out = [[] for i in range(len(dfout))]

        # append to lists if befrate and aftrate are defined and not strings i.e. not "badindex"
        for i in range(len(befrates)):
            for j in range(len(befrates[i])):
                # note that sometimes mats[i][j] can be null
                # for example AUS non-benchmark bond AU3TB0000143 issued on 20120625d but appears in data on 20120619d
                if pd.notnull(befrates[i][j]) and pd.notnull(aftrates[i][j]) and pd.notnull(mats[i][j]) and isinstance(befrates[i][j], str) is False and isinstance(aftrates[i][j], str) is False:
                    befrates_out[i] = befrates_out[i] + [befrates[i][j]]
                    aftrates_out[i] = aftrates_out[i] + [aftrates[i][j]]
                    ids_out[i] = ids_out[i] + [ids[i][j]]
                    mats_out[i] = mats_out[i] + [mats[i][j]]
                    names_out[i] = names_out[i] + [names[i][j]]
                    sources_out[i] = sources_out[i] + [prefix_source]

        dfout['iout__' + prefix_nonsource + '__befrate'] = befrates_out
        dfout['iout__' + prefix_nonsource + '__aftrate'] = aftrates_out
        dfout['iout__' + prefix_nonsource + '__id'] = ids_out
        dfout['iout__' + prefix_nonsource + '__mat'] = mats_out
        dfout['iout__' + prefix_nonsource + '__name'] = names_out
        dfout['iout__' + prefix_nonsource + '__source'] = sources_out
        
    # merge columns together to get iout:}}}

    prefix_nonsources = ['__'.join(col.split('__')[1: -1]) for col in dfout.columns]
    # basic checks and reorder iout:{{{
    for prefix_nonsource in prefix_nonsources:
        befrates = list(dfout['iout__' + prefix_nonsource + '__befrate'])
        aftrates = list(dfout['iout__' + prefix_nonsource + '__aftrate'])
        ids = list(dfout['iout__' + prefix_nonsource + '__id'])
        mats = list(dfout['iout__' + prefix_nonsource + '__mat'])
        names = list(dfout['iout__' + prefix_nonsource + '__name'])
        sources = list(dfout['iout__' + prefix_nonsource + '__source'])

        # now go through and replace rates/maturities accordingly
        for i in range(len(befrates)):
            
            # basic, probably unneeded checks:{{{
            # continue if nan
            if pd.isnull(befrates[i]) is True:
                raise ValueError('befrates is nan when it should not be. Stem: ' + prefix_nonsource + '.')
            if pd.isnull(aftrates[i]) is True:
                raise ValueError('befrates is nan when it should not be. Stem: ' + prefix_nonsource + '.')
            if pd.isnull(mats[i]) is True:
                raise ValueError("befrates and aftrates are not nan but mats is nan. Stem: " + prefix_nonsource + ".")
            if pd.isnull(ids[i]) is True:
                raise ValueError("befrates and aftrates are not nan but ids is nan. Stem: " + prefix_nonsource + ".")
            if pd.isnull(names[i]) is True:
                raise ValueError("befrates and aftrates are not nan but names is nan. Stem: " + prefix_nonsource + ".")
            if pd.isnull(sources[i]) is True:
                raise ValueError("befrates and aftrates are not nan but sources is nan. Stem: " + prefix_nonsource + ".")

            # verify befrates, aftrates, mats have same length
            if len(befrates[i]) != len(aftrates[i]):
                raise ValueError('befrates should have the same length as aftrates. Stem: ' + prefix_nonsource + '. befrates: ' + str(befrates[i]) + '. aftrates: ' + str(aftrates[i]) + '.')
            if len(befrates[i]) != len(mats[i]):
                raise ValueError('befrates should have the same length as mats. Stem: ' + prefix_nonsource + '. befrates: ' + str(befrates[i]) + '. mats: ' + str(mats[i]) + '.')
            if len(befrates[i]) != len(ids[i]):
                raise ValueError('befrates should have the same length as ids. Stem: ' + prefix_nonsource + '. befrates: ' + str(befrates[i]) + '. ids: ' + str(ids[i]) + '.')
            if len(befrates[i]) != len(names[i]):
                raise ValueError('befrates should have the same length as names. Stem: ' + prefix_nonsource + '. befrates: ' + str(befrates[i]) + '. names: ' + str(names[i]) + '.')
            if len(befrates[i]) != len(sources[i]):
                raise ValueError('befrates should have the same length as sources. Stem: ' + prefix_nonsource + '. befrates: ' + str(befrates[i]) + '. names: ' + str(sources[i]) + '.')

            # basic, probably unneeded checks:}}}

            # sort by mats
            matsorder = np.argsort(mats[i])
            befrates[i] = [befrates[i][j] for j in list(matsorder)]
            aftrates[i] = [aftrates[i][j] for j in list(matsorder)]
            mats[i] = [mats[i][j] for j in list(matsorder)]
            ids[i] = [ids[i][j] for j in list(matsorder)]
            names[i] = [names[i][j] for j in list(matsorder)]
            sources[i] = [sources[i][j] for j in list(matsorder)]

        dfout['iout__' + prefix_nonsource + '__befrate'] = befrates
        dfout['iout__' + prefix_nonsource + '__aftrate'] = aftrates
        dfout['iout__' + prefix_nonsource + '__mat'] = mats
        dfout['iout__' + prefix_nonsource + '__id'] = ids
        dfout['iout__' + prefix_nonsource + '__name'] = names
        dfout['iout__' + prefix_nonsource + '__source'] = sources
    # basic checks and reorder iout:}}}

    # remove outliers to get eout:{{{
    # something like ycdi__m1c_1c
    for prefix_nonsource in prefix_nonsources:
        befrates = list(dfout['iout__' + prefix_nonsource + '__befrate'])
        aftrates = list(dfout['iout__' + prefix_nonsource + '__aftrate'])
        ids = list(dfout['iout__' + prefix_nonsource + '__id'])
        mats = list(dfout['iout__' + prefix_nonsource + '__mat'])
        names = list(dfout['iout__' + prefix_nonsource + '__name'])
        sources = list(dfout['iout__' + prefix_nonsource + '__source'])

        # now go through and replace rates/maturities accordingly
        for i in range(len(befrates)):

            # drop i,j if it is an outlier:{{{
            # drop outliers by comparing change in before and after for different maturities
            # need to remove null values beforehand so can compute comparisonchange
            # repeat until no additional outliers found
            oldlen = None
            while len(befrates[i]) != oldlen:
                # record what old length was to verify no more changes made
                oldlen = len(befrates[i])

                keepj = []
                for j in range(len(befrates[i])):
                    # get the size of the change in this point
                    thislevel = befrates[i][j]
                    thischange = aftrates[i][j] - befrates[i][j]

                    thismat = mats[i][j]
                    if thismat < 1.25:
                        lowmat = 0
                        highmat = 2
                    else:
                        lowmat = thismat / 2
                        highmat = thismat * 2

                    # get bonds that I compare j to
                    comparisonks = [k for k in range(len(mats[i])) if mats[i][k] >= lowmat and mats[i][k] <= highmat and k != j]

                    if len(comparisonks) == 0:
                        if thislevel > 20 or thislevel < -2 or abs(thischange) > 0.2:
                            isoutlier = True
                        else:
                            isoutlier = False
                    else:
                        comparisonlevel = np.median([befrates[i][k] for k in comparisonks])
                        comparisonchange = np.median([aftrates[i][k] - befrates[i][k] for k in comparisonks])

                        # now remove outliers by comparing level to comparison level, change to comparison change, and general size of change
                        if abs( 1/(1+thislevel/100) - 1/(1+comparisonlevel/100) ) > 0.05:
                            # I do the adjustment 1/(1+thislevel/100) so that I wouldn't more quickly drop cases where interest rates are very large i.e. difference would need to be 100 and 90 rather than 100 and 95
                            isoutlier = True
                        elif abs(thischange) > 10:
                            isoutlier = True
                        elif abs(thischange) < 0.2:
                            # else if change is small then not an outlier
                            isoutlier = False
                        elif np.sign(thischange) != np.sign(comparisonchange):
                            # if thischange is positive while comparisonchange is negative and more than 0.2p.p. difference probably an outlier
                            isoutlier = True
                        else:
                            # if only a small outlier then only remove if difference from comparisonchange very large
                            if abs(thischange) < 3 * abs(comparisonchange):
                                isoutlier = False
                            else:
                                isoutlier = True
                        # else:
                        #     # if big difference then be less selective about removing outlier
                        #     if abs(thischange) < 3 * abs(comparisonchange):
                        #         isoutlier = False
                        #     else:
                        #         isoutlier = True

                    if isoutlier is False:
                        keepj.append(j)

                # only keep non-outliers
                befrates[i] = [befrates[i][j] for j in keepj]
                aftrates[i] = [aftrates[i][j] for j in keepj]
                mats[i] = [mats[i][j] for j in keepj]
                ids[i] = [ids[i][j] for j in keepj]
                names[i] = [names[i][j] for j in keepj]
                sources[i] = [sources[i][j] for j in keepj]

            # drop i,j if it is an outlier:}}}

            # replace with na if []
            if len(mats[i]) == 0:
                befrates[i] = np.nan
                aftrates[i] = np.nan
                mats[i] = np.nan
                ids[i] = np.nan
                names[i] = np.nan
                sources[i] = np.nan

        dfout['eout__' + prefix_nonsource + '__befrate'] = befrates
        dfout['eout__' + prefix_nonsource + '__aftrate'] = aftrates
        dfout['eout__' + prefix_nonsource + '__mat'] = mats
        dfout['eout__' + prefix_nonsource + '__id'] = ids
        dfout['eout__' + prefix_nonsource + '__name'] = names
        dfout['eout__' + prefix_nonsource + '__source'] = sources
    # remove outliers to get eout:}}}

    dfout = dfout.sort_index(axis = 1)

    return(dfout)


def getindividualbonds(df, unprocessed = False):
    """
    Convert an unprocessed or processed (with all sources) dataframe back to individual bonds
    Rather than lists
    """

    # get list of stems
    befrate_stems = [col[: -9] for col in df.columns if col.endswith('__befrate')]

    # go through befrates/aftrates:{{{
    stemstoconcat = []
    for befrate_stem in befrate_stems:
        befrates = list(df[befrate_stem + '__befrate'])
        aftrates = list(df[befrate_stem + '__aftrate'])
        mats = list(df[befrate_stem + '__mat'])
        ids = list(df[befrate_stem + '__id'])
        names = list(df[befrate_stem + '__name'])
        if unprocessed is True:
            # get sources of same dimension as befrates
            sources = []
            for i in range(len(befrates)):
                sources.append([befrate_stem.split('__')[0]] * len(befrates[i]))
        else:
            sources = list(df[befrate_stem + '__source'])

        rowstoconcat = []

        # now go through and replace rates/maturities accordingly
        for i in range(len(befrates)):
            outdict = {}
            
            # stop if nan or empty list
            if (isinstance(befrates[i], list) is False and pd.isnull(befrates[i]) is True) or (isinstance(befrates[i], list) is True and len(befrates[i]) == 0):
                # add empty row with no columns
                rowstoconcat.append(pd.DataFrame(index = [0]))
                continue

            for j in range(len(befrates[i])):
                if unprocessed is True:
                    outliertype = 'i'
                else:
                    # this is either "e" (excluding outliers) or "i" (including outliers)
                    outliertype = befrate_stem.split('__')[0][0]
                # this is the same for both unprocessed and processed
                ycdi = befrate_stem.split('__')[1]
                if ycdi == 'ycdi':
                    ycpart = 'yc'
                elif ycdi == 'ycdi_il':
                    ycpart = 'yl'
                else:
                    raise ValueError('ycdi part misspecified')
                # this is the same for both unprocessed and processed
                # m1c1c
                timeframe = befrate_stem.split('__')[2]

                # replace z_deu_ref with zdeuref
                source = sources[i][j].replace('_', '')

                first3parts = ycpart + '_' + timeframe + '_' + outliertype + source + '_'
                last2parts = '_na_' + names[i][j]
            
                outdict[first3parts + '0' + last2parts] = [befrates[i][j]]
                outdict[first3parts + '1' + last2parts] = [aftrates[i][j]]
                if unprocessed is True:
                    # strings so sometimes fails
                    try:
                        outdict[first3parts + 'd' + last2parts] = [aftrates[i][j] - befrates[i][j]]
                    except Exception:
                        outdict[first3parts + 'd' + last2parts] = [np.nan]
                else:
                    outdict[first3parts + 'd' + last2parts] = [aftrates[i][j] - befrates[i][j]]
                outdict[first3parts + 'm' + last2parts] = [mats[i][j]]
                outdict[first3parts + 'id' + last2parts] = [ids[i][j]]

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

    
def getbondshocks_yc(dfprocessed, inputlist, printdetails = False):
    """
    Take the output of getbondshocks_process and get yield curves based on inputlist
    inputlist is a list of dictionaries each defining a measure to construct
    """
    if printdetails is True:
        print(str(datetime.datetime.now()) + ' Started getbondshocks_yc.')

    # make deepcopy of inputlist to avoid issues if I reuse the list
    inputlist = copy.deepcopy(inputlist)

    # get list of prefixes for processed bonds
    # for example: eout__ycdi__m1c_1c
    prefixes = sorted(list(set(['__'.join(col.split('__')[0: 3]) for col in dfprocessed.columns])))
    # get bond types
    # for example ['ycdi'], ['ycdi', 'ycdi_il']
    bondtypes = sorted(list(set([col.split('__')[1] for col in dfprocessed.columns])))
    # get bond types
    # for example ['m1c_1c', 'm1c_1o']
    timeframes = sorted(list(set([col.split('__')[2] for col in dfprocessed.columns])))

    nspossibleval = ['ns', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'ns6']
    nsspossibleval = ['nss', 'nss1', 'nss2', 'nss3', 'nss4', 'nss5', 'nss6']

    # adjust input dicts:{{{
    # if inputted single dict, adjust to be a list
    if isinstance(inputlist, dict):
        inputlist = [inputlist]

    # go through inputlist initially
    for i, thisdict in enumerate(inputlist):
        # yctype is na/ns[num/]/wi
        if 'yctype' not in thisdict:
            raise ValueError('yctype must be defined for every inputlist')

        # go through and verify yctype makes sense
        if thisdict['yctype'] == 'na':
            thisdict['nsrank'] = None
        elif thisdict['yctype'] == 'wi':
            thisdict['nsrank'] = None
        elif thisdict['yctype'] in nspossibleval:
            if thisdict['yctype'] == 'ns':
                # use default ns rank
                thisdict['nsrank'] = 1
            else:
                thisdict['nsrank'] = int(thisdict['yctype'][2])
            if 'name' not in thisdict:
                thisdict['name'] = ['m06', 'y01', 'y02', 'y03', 'y04', 'y05', 'y07', 'y10', 'y15']
        elif thisdict['yctype'] in nsspossibleval:
            if thisdict['yctype'] == 'nss':
                # use default ns rank
                thisdict['nsrank'] = 4
            else:
                thisdict['nsrank'] = int(thisdict['yctype'][2])
            if 'name' not in thisdict:
                thisdict['name'] = ['m06', 'y01', 'y02', 'y03', 'y04', 'y05', 'y07', 'y10', 'y15']
        else:
            # yctype should be in one of this list
            raise ValueError('yctype misdefined: ' + thisdict['yctype'] + '.')

        if 'name' not in thisdict:
            if thisdict['yctype'] == 'na':
                raise ValueError('Need to specify name of bonds I want to consider e.g. y01, m06 as "name" in inputlist dict: ' + str(thisdict) + '.')
            elif thisdict['yctype'] == 'wi':
                raise ValueError('Need to specify window of bonds I want to consider e.g. y01y03, m06y02 as "name" in inputlist dict: ' + str(thisdict) + '.')
            else:
                raise ValueError('Need to specify name in inputlist dict: ' + str(thisdict) + '.')

        # ensure name is a list in case I only specified one value
        if isinstance(thisdict['name'], str):
            thisdict['name'] = [thisdict['name']]

        # whether do nominal/inflation-linked or both
        if 'bondtype' not in thisdict:
            thisdict['bondtype'] = bondtypes
        else:
            if isinstance(thisdict['bondtype'], str):
                thisdict['bondtype'] = [thisdict['bondtype']]

            # if specified c replace with ycdi
            # if specified l replace with ycdi_il
            bondtypes2 = []
            for bondtype in thisdict['bondtype']:
                if bondtype == 'c':
                    bondtypes2.append('ycdi')
                elif bondtype == 'l':
                    bondtypes2.append('ycdi_il')
                else:
                    bondtypes2.append(bondtype)
            thisdict['bondtype'] = bondtypes2

            # verify bondtypes specified are available
            for bondtype in thisdict['bondtype']:
                if bondtype not in bondtypes:
                    raise ValueError('Specified bondtype not in list of bondtypes: Specified bond type: ' + bondtype + '. Available bondtypes: ' + str(bondtypes) + '.')

        # which timeframes cover
        if 'timeframe' not in thisdict:
            thisdict['timeframe'] = timeframes
        else:
            if isinstance(thisdict['timeframe'], str):
                thisdict['timeframe'] = [thisdict['timeframe']]
            # go through timeframes in order
            thisdict['timeframe'] = sorted(thisdict['timeframe'])
            # verify that the specified timeframes are correct
            for timeframe in thisdict['timeframe']:
                if timeframe not in timeframes:
                    raise ValueError('Specified timeframe not in list of timeframes: Timeframe: ' + timeframe + '. Timeframes: ' + str(timeframes) + '.')

        # whether exclude/include outliers or do both
        if 'outtype' not in thisdict:
            # this means I exclude outliers
            thisdict['outtype'] = ['e']
        else:
            if isinstance(thisdict['outtype'], str):
                thisdict['outtype'] = [thisdict['outtype']]
            # convert to sorted list
            thisdict['outtype'] = sorted(list(thisdict['outtype']))
            for outtype in thisdict['outtype']:
                if outtype not in ['e', 'i']:
                    raise ValueError('Specified outtype not e/i. Outtype: ' + outtype + '.')

        # which sources cover
        if 'source' not in thisdict:
            # "a" means I'm including all sources
            thisdict['source'] = 'a'

        if 'addinfovars' not in thisdict:
            thisdict['addinfovars'] = False

        # verify no undefined words in thisdict
        for word in thisdict:
            if word not in ['yctype', 'nsrank', 'name', 'bondtype', 'timeframe', 'outtype', 'source', 'addinfovars']:
                raise ValueError('Bad word in ycinputlist dict: ' + word + '.')

        # update thisdict
        inputlist[i] = thisdict

    # adjust input dicts:}}}

    # outdict to which I output dfout columns
    outdict = {}

    # go through columns one at a time
    for thisdict in inputlist:
        if printdetails is True:
            print(str(datetime.datetime.now()) + ' getbondshocks_yc: Started ' + str(thisdict) + '.')
        for bondtype in thisdict['bondtype']:
            for timeframe in thisdict['timeframe']:
                for outtype in thisdict['outtype']:

                    inputprefix = outtype + 'out__' + bondtype + '__' + timeframe

                    # get processed columns as lists
                    aftrates_list = dfprocessed[inputprefix + '__aftrate'].tolist()
                    befrates_list = dfprocessed[inputprefix + '__befrate'].tolist()
                    ids_list = dfprocessed[inputprefix + '__id'].tolist()
                    mats_list = dfprocessed[inputprefix + '__mat'].tolist()
                    names_list = dfprocessed[inputprefix + '__name'].tolist()
                    sources_list = dfprocessed[inputprefix + '__source'].tolist()

                    # get output prefix
                    if bondtype == 'ycdi':
                        bondtype2 = 'c'
                    elif bondtype == 'ycdi_il':
                        bondtype2 = 'l'
                    else:
                        raise ValueError('bondtype should be in ycdi or ycdi_il.')
                    # replace underscores in source i.e. z_deu_ref becomes zdeuref
                    # note I also remove underscores from the processed bond sources below
                    source = thisdict['source'].replace('_', '')
                    # first four parts (plus last hyphen) of output name
                    outputprefix = 'y' + bondtype2 + '_' + timeframe.replace('_', '') + '_' + outtype + source + '_' + thisdict['yctype']

                    # create lists in outdict
                    for name in thisdict['name']:
                        # replace y00_y01 with y00y01
                        name2 = name.replace('_', '')

                        # I always want certain lists
                        outdict[outputprefix + '_0_' + name2] = [np.nan] * len(dfprocessed)
                        outdict[outputprefix + '_1_' + name2] = [np.nan] * len(dfprocessed)
                        outdict[outputprefix + '_d_' + name2] = [np.nan] * len(dfprocessed)

                        # if yctype is na or wi add info by maturity
                        # however I use this very rarely
                        if thisdict['addinfovars'] is True and thisdict['yctype'] in ['na', 'wi']:
                            outdict[outputprefix + '_i_r0_' + name2] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_r1_' + name2] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_i_' + name2] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_m_' + name2] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_n_' + name2] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_s_' + name2] = [np.nan] * len(dfprocessed)
                            
                    # only for ns case - add Nelson-Siegel parameters
                    if thisdict['yctype'] in nspossibleval + nsspossibleval:
                        # always add parameters for ns/nss
                        outdict[outputprefix + '_i_p0'] = [np.nan] * len(dfprocessed)
                        outdict[outputprefix + '_i_p1'] = [np.nan] * len(dfprocessed)
                        outdict[outputprefix + '_i_me'] = [np.nan] * len(dfprocessed)
                        outdict[outputprefix + '_i_ml'] = [np.nan] * len(dfprocessed)

                        # if yctype is na or wi add info for overall bond
                        if thisdict['addinfovars'] is True:
                            outdict[outputprefix + '_i_r0'] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_r1'] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_i'] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_m'] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_n'] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_s'] = [np.nan] * len(dfprocessed)
                            outdict[outputprefix + '_i_rk'] = [np.nan] * len(dfprocessed)

                    # go through one row of columns at a time
                    for i in range(len(dfprocessed)):
                        
                        # selecting elements to use by row:{{{
                        # get single row
                        befrates = befrates_list[i]
                        aftrates = aftrates_list[i]
                        ids = ids_list[i]
                        mats = mats_list[i]
                        names = names_list[i]
                        sources = sources_list[i]

                        # continue if no data for this row
                        if isinstance(befrates, list) is False and pd.isnull(befrates):
                            continue

                        # remove underscores from source to match the adjusted source names
                        sources = [source.replace('_', '') for source in sources]

                        # which points keep generally
                        keepjs = []
                        existingids = []
                        for j in range(len(befrates)):
                            if (thisdict['source'] in ['a', 'b'] or sources[j] == thisdict['source']) and (pd.isnull(ids[j]) or ids[j] not in existingids):
                                keepjs.append(j)
                                existingids.append(ids[j])

                        # adjust lists keeping relevant sources/ids
                        befrates = [befrates[j] for j in keepjs]
                        aftrates = [aftrates[j] for j in keepjs]
                        ids = [ids[j] for j in keepjs]
                        mats = [mats[j] for j in keepjs]
                        names = [names[j] for j in keepjs]
                        sources = [sources[j] for j in keepjs]

                        # stop if list is empty
                        if len(befrates) == 0:
                            continue

                        # keep only best source
                        if thisdict['source'] == 'b':
                            # figure out best source
                            bestsource = None
                            bestrank = None
                            for source in set(sources):
                                matstemp = [mats[j] for j in range(len(sources)) if sources[j] == source]
                                thisrank = getranking(matstemp)
                                if bestrank is None or thisrank > bestrank:
                                    bestsource = source

                            # keep only that source
                            keepjs = []
                            for j in range(len(befrates)):
                                if sources[j] == bestsource:
                                    keepjs.append(j)
                            befrates = [befrates[j] for j in keepjs]
                            aftrates = [aftrates[j] for j in keepjs]
                            ids = [ids[j] for j in keepjs]
                            mats = [mats[j] for j in keepjs]
                            names = [names[j] for j in keepjs]
                            sources = [sources[j] for j in keepjs]

                        # get rank
                        thisrank = getranking(mats)
                        # stop if rank is too low
                        if thisdict['nsrank'] is not None and thisrank < thisdict['nsrank']:
                            continue

                        # selecting elements to use by row:}}}

                        # do this only once - not for every name in thisdict['name']
                        if thisdict['yctype'] in nspossibleval:
                            # getting Nelson-Siegel parameters:{{{
                            # only need to do once for each row

                            success = False
                            try:
                                # note dividing by 100
                                befcurve, befstatus = calibrate_ns_ols(np.array(mats), np.array(befrates) / 100, tau0 = 1.0)
                                aftcurve, aftstatus = calibrate_ns_ols(np.array(mats), np.array(aftrates) / 100, tau0 = 1.0)
                                if befstatus['success'] is True and aftstatus['success'] is True:
                                    success = True
                            except Exception:
                                None

                            """ when NS Luphord algorithm fails{{{
                            A few cases where algorithm does not work properly
                            for example Turkey 20181025d with following parameters:
                            rates: [24.026, 25.63, 24.4, 21.61, 18.28]
                            maturities: [0.5, 0.898, 1.799, 2.91, 9.369]
                            NSLU parameters: [0.24807759929782192, -0.0001242337575754526, -5.045433797156877e-06, -0.3656479693246273]
                            NSLU output: [24.747427383791656, 24.36880120830213, 20.891315860711074, -13.358179418230915, -358.02256000366066, -3765.5866839269547, -35175.06951371506, -274212.8363491629, -1012341.2905774283, 24006077.95079504, 869145377.1650635, 19802051314.59813, 388680603189.31506, 7092446873687.183, 124017502376382.94]
                            issue may be that all but one rate has a maturity of <3 years
                            last NSLU parameter is negative which I think is driving strange numbers so drop if that happens
                            }}}"""

                            # use parameters from algorithm if success is True and parameters are good
                            if success is True:
                                befparamdict = vars(befcurve)
                                aftparamdict = vars(aftcurve)

                                # if tau non-positive or beta1 is very large then set success to be False
                                if befparamdict['tau'] <= 0 or aftparamdict['tau'] <= 0 or abs(befparamdict['beta1']) > 100 or abs(aftparamdict['beta1']) > 100:
                                    success = False

                            # if not worked try with my code setting tau=1
                            if success is False:
                                # use my own code fixing tau parameter to 1
                                # should always run
                                befparams = nsme_singlereg(np.array(mats), np.array(befrates) / 100, 1)
                                aftparams = nsme_singlereg(np.array(mats), np.array(aftrates) / 100, 1)
                                befparamdict = {'beta0': befparams[1], 'beta1': befparams[2], 'beta2': befparams[3], 'tau': befparams[4]}
                                aftparamdict = {'beta0': aftparams[1], 'beta1': aftparams[2], 'beta2': aftparams[3], 'tau': aftparams[4]}

                                if abs(befparamdict['beta1']) <= 100 and abs(aftparamdict['beta1']) <=100:
                                    success = True

                            if success is True:
                                # add params
                                outdict[outputprefix + '_i_p0'][i] = [befparamdict['beta0'], befparamdict['beta1'], befparamdict['beta2'], befparamdict['tau']]
                                outdict[outputprefix + '_i_p1'][i] = [aftparamdict['beta0'], aftparamdict['beta1'], aftparamdict['beta2'], aftparamdict['tau']]

                                # add first/last maturity values
                                outdict[outputprefix + '_i_me'][i] = mats[0]
                                outdict[outputprefix + '_i_ml'][i] = mats[-1]
                            # getting Nelson-Siegel parameters:}}}

                            # save additional NS vars:{{{
                            if thisdict['addinfovars'] is True:
                                outdict[outputprefix + '_i_rk'][i] = thisrank
                                outdict[outputprefix + '_i_r0'][i] = befrates
                                outdict[outputprefix + '_i_r1'][i] = aftrates
                                outdict[outputprefix + '_i_i'][i] = ids
                                outdict[outputprefix + '_i_m'][i] = mats
                                outdict[outputprefix + '_i_n'][i] = names
                                outdict[outputprefix + '_i_s'][i] = sources
                            # save additional NS vars:}}}

                        # do this only once - not for every name in thisdict['name']
                        if thisdict['yctype'] in nsspossibleval:
                            # getting Nelson-Siegel-Svensson parameters:{{{
                            # only need to do once for each row

                            success = False
                            try:
                                # note dividing by 100
                                befcurve, befstatus = calibrate_nss_ols(np.array(mats), np.array(befrates) / 100)
                                aftcurve, aftstatus = calibrate_nss_ols(np.array(mats), np.array(aftrates) / 100)
                                if befstatus['success'] is True and aftstatus['success'] is True:
                                    success = True
                            except Exception:
                                None

                            # use parameters from algorithm if success is True and parameters are good
                            if success is True:
                                befparamdict = vars(befcurve)
                                aftparamdict = vars(aftcurve)

                                # if tau non-positive then set success to be False
                                if befparamdict['tau1'] <= 0 or aftparamdict['tau1'] <= 0 or befparamdict['tau2'] <= 0 or aftparamdict['tau2'] <= 0:
                                    success = False

                            # if not worked try with my code setting tau=1
                            if success is False:
                                # not doing for moment
                                None

                            if success is True:
                                # add params
                                outdict[outputprefix + '_i_p0'][i] = [befparamdict['beta0'], befparamdict['beta1'], befparamdict['beta2'], befparamdict['beta3'], befparamdict['tau1'], befparamdict['tau2']]
                                outdict[outputprefix + '_i_p1'][i] = [aftparamdict['beta0'], aftparamdict['beta1'], aftparamdict['beta2'], aftparamdict['beta3'], aftparamdict['tau1'], aftparamdict['tau2']]

                                # add first/last maturity values
                                outdict[outputprefix + '_i_me'][i] = mats[0]
                                outdict[outputprefix + '_i_ml'][i] = mats[-1]
                            # getting Nelson-Siegel parameters:}}}

                            # save additional NS vars:{{{
                            if thisdict['addinfovars'] is True:
                                outdict[outputprefix + '_i_rk'][i] = thisrank
                                outdict[outputprefix + '_i_r0'][i] = befrates
                                outdict[outputprefix + '_i_r1'][i] = aftrates
                                outdict[outputprefix + '_i_i'][i] = ids
                                outdict[outputprefix + '_i_m'][i] = mats
                                outdict[outputprefix + '_i_n'][i] = names
                                outdict[outputprefix + '_i_s'][i] = sources
                            # save additional NS vars:}}}


                        # get yc
                        for name in thisdict['name']:
                            name2 = name.replace('_', '')

                            if thisdict['yctype'] == 'na':
                                # keep only name starting with this value
                                keepjs = []
                                for j in range(len(befrates)):
                                    if names[j] == name:
                                        keepjs.append(j)
                                befrates2 = [befrates[j] for j in keepjs]
                                aftrates2 = [aftrates[j] for j in keepjs]
                                ids2 = [ids[j] for j in keepjs]
                                mats2 = [mats[j] for j in keepjs]
                                names2 = [names[j] for j in keepjs]
                                sources2 = [sources[j] for j in keepjs]

                                if len(befrates2) == 0:
                                    continue

                                # stop if no values
                                # before/after computation
                                # take mean in case multiple sources
                                before = np.mean(befrates2)
                                after = np.mean(aftrates2)
                                di = after - before

                            elif thisdict['yctype'] in nspossibleval:
                                
                                maturity = getsinglemat(name)

                                # check to ensure not defining bad variables:{{{
                                # Issue with Nelson-Siegel estimation:
                                # - If I only have data for year x onwards, the data pre-year x can be very off
                                # - For example, when I had data for years 2-15 which went from 16-20 (with a bit of a sudden jump), Nelson-Siegel estimated the yield for a 1-year bond to be 14000%
                                # - So should using the estimates for Nelson-Siegel for years for which I do not have the same or a lower maturity bond
                                # - Less of an issue for maturities that exceed my higest maturity because of the way the exponentials in Nelson-Siegel work
                                # - When computing change in yield, drop very high interest rates and large changes as further check
                                # - Note need to adjust which early maturities drop depending on strictness of shock i.e. with s1 can only compute 5+ year change in yield compared to with s5 can compute 1+ year change

                                # don't worry about doing this for ycdi_for where I've already checked the appropriate bonds to include
                                earliestmat = mats[0]
                                if earliestmat > maturity:
                                    continue
                                latestmat = mats[-1]
                                if latestmat < maturity:
                                    continue

                                # check to ensure not defining bad variables:}}}

                                # note multiplying by 100 to return to percentage form
                                befparams = outdict[outputprefix + '_i_p0'][i]
                                aftparams = outdict[outputprefix + '_i_p1'][i]

                                if pd.isnull(befparams) is not True and pd.isnull(aftparams) is not True:
                                    before = nsme_yield(maturity, befparams) * 100
                                    after = nsme_yield(maturity, aftparams) * 100
                                    # before = befcurve(maturity) * 100
                                    # after = aftcurve(maturity) * 100
                                    di = after - before

                            elif thisdict['yctype'] in nsspossibleval:
                                
                                maturity = getsinglemat(name)

                                # check to ensure not defining bad variables:{{{
                                # Issue with Nelson-Siegel estimation:
                                # - If I only have data for year x onwards, the data pre-year x can be very off
                                # - For example, when I had data for years 2-15 which went from 16-20 (with a bit of a sudden jump), Nelson-Siegel estimated the yield for a 1-year bond to be 14000%
                                # - So should using the estimates for Nelson-Siegel for years for which I do not have the same or a lower maturity bond
                                # - Less of an issue for maturities that exceed my higest maturity because of the way the exponentials in Nelson-Siegel work
                                # - When computing change in yield, drop very high interest rates and large changes as further check
                                # - Note need to adjust which early maturities drop depending on strictness of shock i.e. with s1 can only compute 5+ year change in yield compared to with s5 can compute 1+ year change

                                # don't worry about doing this for ycdi_for where I've already checked the appropriate bonds to include
                                earliestmat = mats[0]
                                if earliestmat > maturity:
                                    continue
                                latestmat = mats[-1]
                                if latestmat < maturity:
                                    continue

                                # check to ensure not defining bad variables:}}}

                                # note multiplying by 100 to return to percentage form
                                befparams = outdict[outputprefix + '_i_p0'][i]
                                aftparams = outdict[outputprefix + '_i_p1'][i]

                                if pd.isnull(befparams) is not True and pd.isnull(aftparams) is not True:
                                    befcurve = NelsonSiegelSvenssonCurve(befparams[0], befparams[1], befparams[2], befparams[3], befparams[4], befparams[5])
                                    aftcurve = NelsonSiegelSvenssonCurve(aftparams[0], aftparams[1], aftparams[2], aftparams[3], aftparams[4], aftparams[5])
                                    # before = nsme_yield(maturity, befparams) * 100
                                    # after = nsme_yield(maturity, aftparams) * 100
                                    before = befcurve(maturity) * 100
                                    after = aftcurve(maturity) * 100
                                    di = after - before

                                else:
                                    before = np.nan
                                    after = np.nan
                                    di = np.nan

                            elif thisdict['yctype'] == 'wi':
                                if '_' in name:
                                    # this would work with y01_y03
                                    lowermat = getsinglemat(name.split('_')[0])
                                    uppermat = getsinglemat(name.split('_')[1])
                                else:
                                    # this would work with y01y03
                                    lowermat = getsinglemat(name[0: 3])
                                    uppermat = getsinglemat(name[3: 6])

                                # keep only maturities in window
                                keepjs = []
                                for j in range(len(befrates)):
                                    if mats[j] >= lowermat and mats[j] <= uppermat:
                                        keepjs.append(j)
                                befrates2 = [befrates[j] for j in keepjs]
                                aftrates2 = [aftrates[j] for j in keepjs]
                                ids2 = [ids[j] for j in keepjs]
                                mats2 = [mats[j] for j in keepjs]
                                names2 = [names[j] for j in keepjs]
                                sources2 = [sources[j] for j in keepjs]

                                if len(befrates2) == 0:
                                    continue

                                # stop if no values
                                # before/after computation
                                # take mean in case multiple sources
                                before = np.mean(befrates2)
                                after = np.mean(aftrates2)
                                di = after - before

                            else:
                                raise ValueError('yctype not defined correctly: ' + thisdict['yctype'] + '.')

                            # add to lists
                            outdict[outputprefix + '_0_' + name2][i] = before
                            outdict[outputprefix + '_1_' + name2][i] = after
                            outdict[outputprefix + '_d_' + name2][i] = di

                            if thisdict['yctype'] in ['na', 'wi'] and thisdict['addinfovars'] is True:
                                outdict[outputprefix + '_i_r0_' + name2][i] = befrates2
                                outdict[outputprefix + '_i_r1_' + name2][i] = aftrates2
                                outdict[outputprefix + '_i_i_' + name2][i] = ids2
                                outdict[outputprefix + '_i_m_' + name2][i] = mats2
                                outdict[outputprefix + '_i_n_' + name2][i] = names2
                                outdict[outputprefix + '_i_s_' + name2][i] = sources2

    # combine into dataframe
    dfout = pd.DataFrame(outdict, dfprocessed.index)

    dfout = dfout.sort_index(axis = 1)

    if printdetails is True:
        print(str(datetime.datetime.now()) + ' Finished getbondshocks_yc.')

    return(dfout)


# Post-Processing Functions:{{{1
def getaltnsmat(df, ycstem, maturity, includeoutsidevals = False):
    """
    Take the parameters from the Nelson-Siegel/Nelson-Siegel-Svensson approach and construct before/after/difference values for alternative maturities

    df is the dataframe
    ycstem is the first four parts i.e. yc_30m_ea_ns
    maturity is y09
    includeoutsidevals is False means I set dates where the earliest/latest maturity of the fixed income instruments available is higher/lower than the maturity I am computing to be nan (in other words I'm computing a maturity that's outside the range of the fixed income instruments I have values for)

    getaltnsmat(df, 'yc_30m_ea_ns', 'y09') would create yc_30m_ea_ns_0_y09, yc_30m_ea_ns_1_y09, yc_30m_ea_ns_d_y09 using yc_30m_ea_ns_i_p0 and yc_30m_ea_ns_i_p1
    """

    if ycstem + '_i_p0' not in df.columns:
        raise ValueError('getaltnsmat requires ' + ycstem + '_i_p0 to be defined')
    if ycstem + '_i_p1' not in df.columns:
        raise ValueError('getaltnsmat requires ' + ycstem + '_i_p1 to be defined')

    maturityval = getsinglemat(maturity)

    befcurves = list(df[ycstem + '_i_p0'])
    aftcurves = list(df[ycstem + '_i_p1'])
    earliestmats = list(df[ycstem + '_i_me'])
    latestmats = list(df[ycstem + '_i_ml'])

    befvals = [np.nan] * len(befcurves)
    aftvals = [np.nan] * len(befcurves)
    divals = [np.nan] * len(befcurves)

    for i in range(len(befcurves)):
        befcurve = befcurves[i]
        aftcurve = aftcurves[i]
        earliestmat = earliestmats[i]
        latestmat = latestmats[i]

        # evaluate curves to convert from string to list
        if isinstance(befcurve, str):
            befcurve = eval(befcurve)
        if isinstance(aftcurve, str):
            aftcurve = eval(aftcurve)

        if (not isinstance(befcurve, list) and pd.isnull(befcurve)) or (not isinstance(aftcurve, list) and pd.isnull(aftcurve)):
            continue
        # drop maturity values which are outside the range of fixed income data I have available
        if  includeoutsidevals is False and (maturityval < earliestmat or maturityval > latestmat):
            continue

        # convert back into curves (rather than just parameters)
        if len(befcurve) == 4:
            befcurve = NelsonSiegelCurve(befcurve[0], befcurve[1], befcurve[2], befcurve[3])
            aftcurve = NelsonSiegelCurve(aftcurve[0], aftcurve[1], aftcurve[2], aftcurve[3])
        elif len(befcurve) == 6:
            befcurve = NelsonSiegelSvenssonCurve(befcurve[0], befcurve[1], befcurve[2], befcurve[3], befcurve[4], befcurve[5])
            aftcurve = NelsonSiegelSvenssonCurve(aftcurve[0], aftcurve[1], aftcurve[2], aftcurve[3], aftcurve[4], aftcurve[5])
        else:
            raise ValueError('Some error')

        # get values for relevant maturities
        # note multiplying by 100 to return to percentage form
        before = befcurve(maturityval) * 100
        after = aftcurve(maturityval) * 100
        di = after - before

        # adjust
        befvals[i] = before
        aftvals[i] = after 
        divals[i] = di 

    df[ycstem + '_0_' + maturity] = befvals
    df[ycstem + '_1_' + maturity] = aftvals
    df[ycstem + '_d_' + maturity] = divals


def getaltnsmat_varname(df, varnames, includeoutsidevals = False):
    """
    This does getaltnsmat but where I input a list of varnames of single varname
    Note that I can input yc_30m_ea_ns_0_y03, yc_30m_ea_ns_1_y03, yc_30m_ea_ns_d_y03 and in all cases I will create all three variables
    """
    
    if isinstance(varnames, str):
        varnames = [varnames]

    for varname in varnames:
        if varname in df:
            continue
        ycstem = '_'.join(varname.split('_')[0: 4])
        maturity = varname.split('_')[5]
        getaltnsmat(df, ycstem, maturity, includeoutsidevals=includeoutsidevals)

    return(df)


def forwarddiff(df, prefix, start, end):
    """
    Returns the annual change in the forward part of the curve

    Input:
    - dataset
    - prefix: yc_20m_ea_ns
    - ycnamestart: y01 (for ns), y00y02 (for wi)
    - ycnameend: y11 (for ns), y07y13 (for wi)
    Output:
    - yc_20m_ea_ns_fd_y01_y11 or yc_1h_ea_wi_fd_y00y02_y07y13

    Structure of difference variable
    ycdi__m1h_1h__fdi__YCNAME1__YCNAME2
    prefix = 'ycdi__m1h_1h'
    ycnamestart = 'ridge__y01_y03'
    ycnameend = 'nslu__y05'
    """

    yctype = prefix.split('_')[3]
    if yctype not in ['ns', 'wi']:
        raise ValueError('yctype misspecified: ' + str(yctype) + '.')

    # verify ycnamestart and ycnameend have correct numbers of letters
    if yctype == 'wi':
        if len(start) != 6:
            raise ValueError('For wi, start should have 6 letters: ' + start + '.')
        if len(end) != 6:
            raise ValueError('For wi, end should have 6 letters: ' + end + '.')
    elif yctype == 'ns':
        if len(start) != 3:
            raise ValueError('For ns, start should have 6 letters: ' + start + '.')
        if len(end) != 3:
            raise ValueError('For wi, end should have 3 letters: ' + end + '.')
    else:
        raise ValueError('yctype misspecified: ' + yctype + '.')

    # get maturity of start
    if yctype == 'wi':
        matlowpart = start[0: 3]
        mathighpart = start[3: 6]

        matlow = getsinglemat(matlowpart)
        mathigh = getsinglemat(mathighpart)

        matstart = (matlow + mathigh) / 2
    elif yctype == 'ns':
        matstart = getsinglemat(start)
    else:
        raise ValueError('yctype misspecified: ' + yctype + '.')

    # get maturity of end
    if yctype == 'wi':
        matlowpart = end[0: 3]
        mathighpart = end[3: 6]

        matlow = getsinglemat(matlowpart)
        mathigh = getsinglemat(mathighpart)

        matend = (matlow + mathigh) / 2
    elif yctype == 'ns':
        matend = getsinglemat(end)
    else:
        raise ValueError('yctype misspecified: ' + yctype + '.')

    if prefix + '_d_' + start not in df.columns:
        raise ValueError('Missing: ' + prefix + '_d_' + start + '.')
    if prefix + '_d_' + end not in df.columns:
        raise ValueError('Missing: ' + prefix + '_d_' + end + '.')


    df[prefix + '_fd_' + start + '_' + end] = (matend * df[prefix + '_d_' + end] - matstart * df[prefix + '_d_' + start]) / (matend - matstart)
    
    return(df)


def forwarddiff_varname(df, varnames):
    if isinstance(varnames, str):
        varnames = [varnames]

    for varname in varnames:
        if varname in df:
            continue
        prefix = '_'.join(varname.split('_')[0: 4])
        if varname.split('_')[4] != 'fd':
            raise ValueError('Fifth part of varname should be fd for forwarddif_varname.')
        ycnamestart = varname.split('_')[5]
        ycnameend = varname.split('_')[6]
        df = forwarddiff(df, prefix, ycnamestart, ycnameend)

    return(df)


def forwarddiff_test():
    df = pd.DataFrame({'yc_m1h1h_ea_wi_d_y00y02': [0.1, 0.2], 'yc_m1h1h_ea_wi_d_y01y03': [0.2, 0.3]})
    df = forwarddiff(df, 'yc_m1h1h_ea_wi', 'y00y02', 'y01y03')
    """
    Should find 0.3 for first row and 0.4 for second row
    (0.2 * 2 - 0.1 * 1) / (2 - 1) = 0.3
    (0.3 * 2 - 0.2 * 1) / (2 - 1) = 0.4
    """
    df['ans'] = [0.3, 0.4]
    df2 = df[df['yc_m1h1h_ea_wi_fd_y00y02_y01y03'] - df['ans'] > 1e-5]
    if len(df2) > 0:
        print(df2)
        raise ValueError('forwarddiff_test failed')

    # same but with the _varname version of the function
    df = pd.DataFrame({'yc_m1h1h_ea_wi_d_y00y02': [0.1, 0.2], 'yc_m1h1h_ea_wi_d_y01y03': [0.2, 0.3]})
    df = forwarddiff_varname(df, 'yc_m1h1h_ea_wi_fd_y00y02_y01y03')
    """
    Should find 0.3 for first row and 0.4 for second row
    (0.2 * 2 - 0.1 * 1) / (2 - 1) = 0.3
    (0.3 * 2 - 0.2 * 1) / (2 - 1) = 0.4
    """
    df['ans'] = [0.3, 0.4]
    df2 = df[df['yc_m1h1h_ea_wi_fd_y00y02_y01y03'] - df['ans'] > 1e-5]
    if len(df2) > 0:
        print(df2)
        raise ValueError('forwarddiff_test failed')


# Long to Wide Interest Rates for Many Zones/Same Dates:{{{1
def getdomestic_fromlong(dflong, dfdates, datevar, suffix = 'd'):
    """
    dfdates gives a list of the dates associated with each zone, so it contains a 'zone' and a datevar variable
    We then restrict dflong to only the dates associated with each zone
    """

    dfdomestic = dflong.merge(dfdates[['zone', datevar]], on = ['zone', datevar], how = 'right')

    # rename
    dfdomestic = dfdomestic.set_index(['zone', datevar])
    dfdomestic = dfdomestic.rename({col: col.split('_')[0] + suffix + '_' + '_'.join(col.split('_')[1: ]) for col in dfdomestic.columns}, axis = 1, errors = 'raise')

    return(dfdomestic)


def getave_fromlong(dflong, dfdates, datevar, zonesinclude = None, suffix = ''):
    """
    This computes the global mean including the local zone for variables around the mean
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
    # for zone in zones:
    #     dfwide.loc[dfwide['zone'] == zone, [col for col in dfwide.columns if col.split('__')[-1] == zone]] = np.nan

    # FOR EACH ZONE
    dfglobal = pd.DataFrame({'zone': dfwide['zone'], datevar: dfwide[datevar]})
    for varname in varnames:
        # get all columns relating to a given varname
        dfwide2 = dfwide[[col for col in dfwide.columns if '__'.join(col.split('__')[: -1]) == varname]]
        # define a single global varname for that variable
        dfglobal[varname.split('_')[0] + suffix + '_' + '_'.join(varname.split('_')[1: ])] = dfwide2.mean(axis = 1)

    return(dfglobal)


def getforeign_fromlong(dflong, dfdates, datevar, zonesinclude = None, suffix = 'f'):
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
        dfglobal[varname.split('_')[0] + suffix + '_' + '_'.join(varname.split('_')[1: ])] = dfwide2.mean(axis = 1)

    return(dfglobal)


def getzones_fromlong(dflong, datevar, dfdates = None, zonesinclude = None, suffix = 'z'):
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

    # add suffix to first part of name
    if suffix is not None:
        dfwide.columns = [col[0].split('_')[0] + 'z' + col[1].lower() + '_' + '_'.join(col[0].split('_')[1: ]) for col in dfwide.columns]

    # if there are object dtypes in pandas pivot seems to make all variables object afterwards
    # therefore convert back to numeric where possible
    # dfwide = dfwide.astype(float, errors = 'ignore')
    for var in dfwide.columns:
        dfwide[var] = pd.to_numeric(dfwide[var])

    # get index back
    dfwide = dfwide.reset_index()

    if dfdates is not None:
        dfwide = dfwide.merge(dfdates[['zone', datevar]], on = datevar, how = 'right')

    return(dfwide)


# Overall:{{{1
def testall():
    forwarddiff_test()


# Run:{{{1
if __name__ == "__main__":
    testall()
