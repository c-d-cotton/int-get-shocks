#!/usr/bin/env python3
"""
Functions to get surprises around events
"""

relativetoprojectdir = '/'

# preamble_macrodata:{{{
# DO NOT CHANGE BETWEEN the two lines beginning # preamble_macrodata
import os
from pathlib import Path
import sys

try:
    __projectdir__ = Path(os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + relativetoprojectdir))
except NameError:
    __projectdir__ = Path(os.path.abspath("") + relativetoprojectdir)

rootdirpath = __projectdir__ / Path('../mdexternalpath.txt')
homedirpath = Path.home() / Path('.mdexternalpath.txt')
if os.path.isfile(rootdirpath):
    macrodataexternal_readfile = rootdirpath
elif os.path.isfile(homedirpath):
    macrodataexternal_readfile = homedirpath
else:
    print('Warning: macrodata-external folder cannot be found. Maybe need to open T:/ drive on Windows.')

# read location of macrodata-external folder from file
with open(macrodataexternal_readfile) as f:
    macrodataexternal = f.read()
if macrodataexternal[-1] == '\n':
    macrodataexternal = macrodataexternal[: -1]
macrodataexternal = Path(macrodataexternal)

# DO NOT CHANGE BETWEEN the two lines beginning # preamble_macrodata
# preamble_macrodata:}}}

import datetime
import holidays
import numpy as np
import pandas as pd

sys.path.append(str(__projectdir__ / Path('submodules/time-index-func')))
from time_index_func import *

# Daily Data Functions:{{{1
def getrelativedates(eventdates, rday, weekendignore = False, holidayignore = False, startofday = False):
    """
    Get dates relative to another set of dates.
    So if rday is -1 get the dates one day before eventdates

    If weekendignore is True, ignore weekends so return a Friday one day before a Monday
    Can also specify weekendignore = 'ISR' which will exclude weekends by zone. For Israel, this would remove Fridays/Saturdays.

    If holidayignore = 'JPN', remove the Japanese holidays

    If day is an ignored day and rday == 0 then set it to be 'drop'

    startofday is only used when weekendignore is True and rday is 0. Then it tells me whether to take the next day or the prior day
    """

    # adjust arguments:{{{
    if weekendignore is False:
        weekendignore2 = None
    elif weekendignore is True:
        weekendignore2 = 'satsun'
    elif isinstance(weekendignore, str):
        # verify matches zone code
        if len(weekendignore) != 3 or weekendignore.upper() != weekendignore:
            raise ValueError('weekendignore is a string but not capitalized with 3 digits: ' + singledict['weekendignore'] + '.')
        # get list of zones with different weekends
        weekends_frisat = get_weekend_fri_sat()
        if weekendignore in weekends_frisat:
            weekendignore2 = 'frisat'
        else:
            weekendignore2 = 'satsun'
    else:
        raise ValueError('weekendignore is misspecified.')

    if holidayignore is not False:
        # then holidayignore should be a zone - verify fits pattern
        if len(holidayignore) != 3 or holidayignore.upper() != holidayignore:
            raise ValueError('holidayignore is not False but is also not a capitalized string with 3 digits: ' + holidayignore + '.')
        try:
            holidayignore2 = holidays.CountryHoliday(holidayignore)
        except Exception:
            # just don't set holidayignore2 if can't creat dict using holidays
            holidayignore2 = False
    else:
        holidayignore2 = False
    # adjust arguments:}}}

    # get dates:{{{
    relativedates = []
    for date in eventdates:
        dt = convertmytimetodatetime(date)
        # add rday amount to shockday
        dt = dt + datetime.timedelta(rday)

        # now iterate until get right day
        i = 0
        while i < 28:
            badday = False
            weekday = dt.weekday()
            if weekendignore2 == 'satsun' and weekday in [5, 6]:
                badday = True
            if weekendignore2 == 'frisat' and weekday in [4, 5]:
                badday = True
            if holidayignore2 is not False:
                # sometimes holidays package returns NotImplementedError when date is outside possible range
                # in this case don't set it to be a badday
                try:
                     if dt in holidayignore2:
                        badday = True
                except NotImplementedError:
                    None
            if badday is False:
                break
            else:
                # raise counter
                i += 1
                if i >= 28:
                    raise ValueError('Cannot find a day that satisfies weekendignore and holidayignore.')
                # if rday is positive then add a day otherwise subtract a day
                if rday > 0 or (rday == 0 and startofday is False):
                    dt = dt + datetime.timedelta(days = 1)
                elif rday < 0 or (rday == 0 and startofday is True):
                    dt = dt - datetime.timedelta(days = 1)
                else:
                    raise ValueError('Something has gone wrong!')

        # convert back to usual format
        mytime = convertdatetimetomytime(dt, 'd')
        relativedates.append(mytime)
    # get dates:}}}

    return(relativedates)


def getrelativedates_test():
    """
    Monday 29th November 2021
    Sunday 28th November 2021
    Thursday 25th November 2021 was Thanksgiving in the USA
    """

    # main weekend/holiday ignore tests:
    print('20211128d = ' + getrelativedates(['20211129d'], -1)[0])
    print('20211126d = ' + getrelativedates(['20211129d'], -1, weekendignore = 'USA')[0])
    print('20211128d = ' + getrelativedates(['20211129d'], -1, weekendignore = 'ISR')[0])
    print('20211125d = ' + getrelativedates(['20211126d'], -1, holidayignore = False)[0])
    print('20211124d = ' + getrelativedates(['20211126d'], -1, holidayignore = 'USA')[0])

    # if weekend and rday == 0 then should drop
    print('dropped = ' + getrelativedates(['20211128d'], 0, weekendignore = True)[0])


def getdfdailyrel_single(df, eventdates, rday, weekendignore = False, holidayignore = False, ffill = False, bfill = False, ffillfirst = False, eventtimes = None, eventtimebefore = False, eventtimeafter = False, startofday = False):
    """

    Main inputs:
    -   df is daily data over which I get the relative dates
        The index of daily should be in standard form i.e. 20210304d
        df CAN have gaps since I search by loc rather than iloc
        EXCEPTION: df cannot have gaps if I use ffill/bfill since when I select 90 days before that day may not exist in the data even though it should have been filled in
    -   eventdates is a list of dates in the format 20210304d on which the events occurred
    -   rday: The relative days to eventdates that I'm getting
    -   weekendignore/holidayignore same as in getrelativedates
    -   ffill/bfill means I fill the df forward by ffill/bfill (unless ffill/bfill is False)

    Time inputs:
    eventtimebefore and eventtimeafter allow me to require that the eventtime was before or after some time
    Useful particularly when rday is 0 because markets may only close at 4pm so any event taking place at 5pm would not be captured in the close data
    Same with open data when may want data to take place after some time so captured in the open
    If specify eventtimebefore or eventtimeafter then must specify eventtimes
    eventtimes format: List same length as eventdates
    If eventtime is not known then can specify np.nan and will be ignored
    eventtimebefore and eventtimeafter in format 1410M
    eventtimes in format either 1410M or 20210101_1410M
    ffillfirst specifies whether do ffill or bfill first if do both
    """
    if ffillfirst is True:
        if ffill is not False:
            df = df.ffill(limit = ffill)
        if bfill is not False:
            df = df.bfill(limit = bfill)
    else:
        if bfill is not False:
            df = df.bfill(limit = bfill)
        if ffill is not False:
            df = df.ffill(limit = ffill)
        

    # get relativedays
    relativedates = getrelativedates(eventdates, rday, weekendignore = weekendignore, holidayignore = holidayignore, startofday = startofday)


    # get which ones are available
    dfindex = list(df.index)
    relativedates_available_bool = [True if relativedate in dfindex else False for relativedate in relativedates]
    eventdates_available = [eventdates[i] for i in range(len(eventdates)) if relativedates_available_bool[i] is True]
    relativedates_available = [relativedates[i] for i in range(len(relativedates)) if relativedates_available_bool[i] is True]

    # get df for available times
    # this is the step that takes time
    dfshock = df.loc[relativedates_available]

    # drop any duplicates before reindexing (otherwise won't work)
    dfshock = dfshock.groupby(dfshock.index).first()
    # get all relativedates
    dfshock = dfshock.reindex(relativedates)

    # set index to be eventdates
    dfshock.index = eventdates

    # # Old:{{{
    # # get days that are available in the dataset
    # dfindex = list(df.index)
    # relativedates_available = []
    # relativedates_available_index = []
    # for i, relativedate in enumerate(relativedates):
    #     if relativedate in dfindex:
    #         relativedates_available.append(relativedate)
    #         relativedates_available_index.append(i)
    # # restrict only to days available
    # dfshock = df.loc[relativedates_available]
    #
    # # want to add back in relativedates I dropped as they weren't available
    # # can't just reindex as relativedates may contain duplicates (since 1 business day after Saturday/Sunday is Monday)
    # # first replace dfshock index with relativedates indexes that are available
    # dfshock.index = relativedates_available_index
    #
    # # now reindex so covers all potential indexes for relative dates
    # dfshock = dfshock.reindex(list(range(len(relativedates))))
    #
    # # adjust relativedates so remove 'dropped'
    # # this helps to avoid issues where I try to reindex dfmaturity by relativedates but cannot due tot he 'dropped' indexes
    # # 'dropped' only occurs when rday == 0, weekendignore == True and the eventday is a weekend
    # # df will be na for these rows in any case after the reindexing
    # relativedates = [eventdates[i] if relativedates[i] == 'dropped' else relativedates[i] for i in range(len(relativedates))]
    #
    # # now replace back with original relativedates
    # dfshock.index = relativedates
    # # Old:}}}

    # adjust for eventtime:{{{
    # remove if time is not before a given time or after a given time (or is na)
    if eventtimebefore is not False or eventtimeafter is not False:
        # verify eventtimes are defined
        if eventtimes is None:
            raise ValueError('eventtimes must not be None since we specified we want times to be before/after some time.')
        # verify eventtimes are defined correctly
        for eventtime in eventtimes:
            if not pd.isnull(eventtime) and not eventtime.endswith('M'):
                raise ValueError('eventtime is misspecified: ' + str(eventtime) + '.')

    if eventtimebefore is not False:
        # verify eventtimebefore satisfy standard format
        try:
            int(eventtimebefore[0: 4])
        except Exception:
            raise ValueError('eventtimebefore is misspecified: ' + str(eventtimebefore) + '.')

        # remove if eventtime not before eventtimebefore
        exclude = [True if pd.isnull(eventtime) or eventtime[-5: ] > eventtimebefore else False for eventtime in eventtimes]
        dfshock.loc[exclude, :] = np.nan

    if eventtimeafter is not False:
        # verify eventtimeafter satisfy standard format
        try:
            int(eventtimeafter[0: 4])
        except Exception:
            raise ValueError('eventtimeafter is misspecified: ' + str(eventtimeafter) + '.')
        # remove if eventtime not after eventtimeafter
        exclude = [True if pd.isnull(eventtime) or eventtime[-5: ] < eventtimeafter else False for eventtime in eventtimes]
        dfshock.loc[exclude, :] = np.nan
    # adjust for eventtime:}}}

    return(dfshock, relativedates)


def getdfdailyrel_single_test1():
    dfdaily = pd.read_csv(macrodataexternal / Path('int-stock-daily/output/merge/USA.csv'), index_col = 0)

    eventdays = [str(year) + '0115d' for year in range(1990, 2020)]

    dfdailyrel, relativedates = getdfdailyrel_single(dfdaily, eventdays, -1, weekendignore = True)
    print(dfdailyrel)
    print(relativedates)


def getdfdailyrel_single_test2():
    """
    Here we test some special cases.

    Sunday September 14th was day before Lehman Brothers bankruptcy
    """
    dfdaily = pd.read_csv(macrodataexternal / Path('int-stock-daily/output/merge/AUS.csv'), index_col = 0)
    # keep only Australian stock
    dfdaily = dfdaily[["aus__ref__sto__axjo"]]

    # this is a case where the eventtime is before eventtimebefore so should not be skipped
    dfdailyrel, relativedates = getdfdailyrel_single(dfdaily, ["20080915d"], 1, eventtimes = ["1200M"], eventtimebefore = "1600M")
    print("Not na: " + str(dfdailyrel.values[0, 0]))
    # this is a case where the eventtime is after eventtimebefore so should be skipped
    dfdailyrel, relativedates = getdfdailyrel_single(dfdaily, ["20080915d"], 1, eventtimes = ["1700M"], eventtimebefore = "1600M")
    print("Na: " + str(dfdailyrel.values[0, 0]))


def fillgaps_days(df):
    firstday = df.index[0]
    lastday = df.index[-1]
    firstday_dt = convertmytimetodatetime(firstday)
    lastday_dt = convertmytimetodatetime(lastday)

    dates_dt = pd.date_range(start = firstday_dt, end = lastday_dt).to_pydatetime()
    dates_mytime = [convertdatetimetomytime(date_dt, 'd') for date_dt in dates_dt]

    df = df.reindex(dates_mytime)

    return(df)


def getdfdailyrel_reldict(dfclose, eventdates, reldict, dfopen = None, eventtimes = None, holidayignore = None, weekendignore = None, fillgaps = True, zone = None):
    """{{{
    dfclose is daily data over which I get the relative dates
    The index of daily should be in standard form i.e. 20210304d
    dfclose CAN have gaps since I search by loc rather than iloc
    EXCEPTION: If use ffill/bfill then need to fill gaps

    eventdates is a list of dates in the format 20210304d on which the events occurred

    reldict is a dictionary over all the shocks I want to create.
    So inputs will be something like 'm1c', 'm1o' etc.
    This will return a dictionary of arguments
    i.e. reldict['m1c'] = {'rday': 0, 'weekendignore': False, 'holidayignore': False, 'open': False, 'ffill': False, 'bfill': False, 'eventtimebefore': False, 'eventtimeafter': False}
    rday: The day relative to the shock so if rday is -1 then take the day before the shockday
    weekendignore: If the shockday is Monday and rday is -1 then use Friday if weekendignore is True
    holidayignore: Ignore holidays when getting the relative days if holidayignore is True
    open: Use open data rather than closed data
    ffill: If not False then specify an integer for number of days to fill data forward when computing shocks
    bfill: equivalent to ffill but backwards
    zone: Only used when holidayignore = True (since need to specify which zone holidays coming from)


    dfopen is the df with open data rather than closed data
    If it's not specified then dfclose could really be any data.

    eventtimes are a list of the eventtimes either given as '1010M' or np.nan if not available

    Minor options:
    - I also allow weekendignore/holidayignore inputs to the function in case I want to vary this by zone.
    - fillgaps: Automatically sort and fill gaps in data. Filling gaps only needed with ffill/bfill. No cost to doing this other than time. Can turn off if I want to save time though I think it's probably not going to take long in general.
    - ffillfirst: If both ffill and bfill are specified do ffill first if this is True (default is False)
    }}}"""

    # basic checks on reldict:{{{
    for name in reldict:
        if 'dfrel' in reldict[name]:
            raise ValueError('Need to work with reldict where I have not already applied getdfdailyrel_reldict i.e. by applying copy.deepcopy to reldict beforehand.')

        singledict = reldict[name]

        # add defaults from function inputs
        # I adjust holidayignore to be the relevant zone later
        if holidayignore is not None:
            singledict['holidayignore'] = holidayignore
        if weekendignore is not None:
            singledict['weekendignore'] = weekendignore

        # go through and check arguments to singledict
        if 'rday' not in singledict:
            raise ValueError('rday not specified in reldict')
        if 'weekendignore' not in singledict:
            singledict['weekendignore'] = False
        if 'holidayignore' not in singledict:
            singledict['holidayignore'] = False
        if 'open' not in singledict:
            singledict['open'] = False
        if 'ffill' not in singledict:
            singledict['ffill'] = False
        if 'bfill' not in singledict:
            singledict['bfill'] = False
        if 'ffillfirst' not in singledict:
            singledict['ffillfirst'] = False
        if 'eventtimebefore' not in singledict:
            singledict['eventtimebefore'] = False
        if 'eventtimeafter' not in singledict:
            singledict['eventtimeafter'] = False

        # add in specified eventtimes
        singledict['eventtimes'] = eventtimes
        
        if singledict['open'] is True:
            if dfopen is None:
                raise ValueError('Open option specified but no dfopen included.')

        # adjust holidayignore to be local zone
        if singledict['holidayignore'] is True:
            if zone is not None:
                singledict['holidayignore'] = zone
            else:
                raise ValueError('When specify holidayignore is True, need to specify zone when calling getdfdailyrel_reldict (to specify relevant zone for holidays.')
    # basic checks on reldict:}}}

    if fillgaps is True:
        # basic checks on dataframe
        # sort to ensure m1c points to previous day
        dfclose = dfclose.sort_index()
        # fill in gaps to ensure that ffill/bfill work correctly (not needed otherwise)
        dfclose = fillgaps_days(dfclose)
        if dfopen is not None:
            dfopen = dfopen.sort_index()
            dfopen = fillgaps_days(dfopen)

    for shockname in reldict:
        if reldict[shockname]['open'] is True:
            df = dfopen
        else:
            df = dfclose

        dfrel, relativedates = getdfdailyrel_single(df, eventdates, reldict[shockname]['rday'], weekendignore = reldict[shockname]['weekendignore'], holidayignore = reldict[shockname]['holidayignore'], ffill = reldict[shockname]['ffill'], bfill = reldict[shockname]['bfill'], ffillfirst = reldict[shockname]['ffillfirst'], eventtimes = reldict[shockname]['eventtimes'], eventtimebefore = reldict[shockname]['eventtimebefore'], eventtimeafter = reldict[shockname]['eventtimeafter'], startofday = reldict[shockname]['open'])

        reldict[shockname]['dfrel'] = dfrel
        reldict[shockname]['relativedates'] = relativedates

    return(reldict)


def getdfdailyrel_reldict_test(printdetails = False):
    df = pd.read_csv(macrodataexternal / Path('int-stock-daily/output/merge/USA.csv'), index_col = 0)

    # separate df into open and closed
    openvars = [col for col in df.columns if col.split('__')[2].endswith('_op')]
    dfclose = df[[col for col in df.columns if col not in openvars]]
    dfopen = df[openvars]
    # rename dfopen
    dfopen.columns = [col.replace('_op__', '__') for col in dfopen.columns]

    # Lehman Brothers bankruptcy
    eventdates = ['20080915d']

    # get close 1 business day before and open 1 business day after
    reldict = {'m1c': {'rday': -1, 'weekendignore': True}, '1o': {'rday': 1, 'weekendignore': True, 'open': True}}

    # get all daily relative shocks i.e. -1 and +1 days
    reldict = getdfdailyrel_reldict(dfclose, eventdates, reldict, dfopen = dfopen)

    if printdetails is True:
        print(reldict['m1c']['dfrel'])
        print(reldict['m1c']['relativedates'])
        print(reldict['1o']['dfrel'])
        print(reldict['1o']['relativedates'])

    return(reldict)


# Intraday Data Functions:{{{1
def roundnearestinterval(eventtimes, minutetoroundto, minutesadd = 0):
    """
    This rounds a list of eventtimes to the nearest ``minutetoroundto'' i.e. nearest 5 mins
    I may also add minutes to the amount using minutesadd. I can use this to take floors or ceilings For example:
    - If I add 2 minutes then 15:56 becomes 15:58 and will then round to 16:00
    - If I add -2 minutes then 15:59 becomes 15:57 and will then round to 15:55
    """

    eventtimes_rounded = [np.nan] * len(eventtimes)
    for i in range(len(eventtimes)):
        eventtime = eventtimes[i]

        if pd.isnull(eventtime) is True:
            continue

        dt = convertmytimetodatetime(eventtime)
        # add half the amount I'm rounding to (for example 2.5 minutes in the case of 5 minutes)
        # this means that when take the floor, 15:59 will become 16:01.5 and rounds down to 16:00
        # also allow for the possibility of adding minutesadd
        dt = dt + datetime.timedelta(minutes = minutetoroundto / 2 + minutesadd)
        # take the floor of the minutes component
        dt = datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute // minutetoroundto * minutetoroundto)
        mytime = convertdatetimetomytime(dt, 'M')

        eventtimes_rounded[i] = mytime

    return(eventtimes_rounded)


def fillinminsdf(df, interval_minutes):
    """
    df has an index in my usual format i.e. 20210304_1015M
    mins is the number of minutes between points. This can be more than 60 minutes.

    This function then fills in any gaps in the df's index
    """
    startdt = convertmytimetodatetime(df.index[0])
    enddt = convertmytimetodatetime(df.index[-1])
    dt = startdt
    dt_list = []
    while dt <= enddt:
        dt_list.append(dt)
        dt = dt + datetime.timedelta(minutes = interval_minutes)
    mytime_list = [convertdatetimetomytime(dt, 'M') for dt in dt_list]

    df = df.reindex(mytime_list)

    return(df)


def fillinminsdf_test():
    df = pd.read_csv(macrodataexternal / Path('int-bond-intra/output/merge/bond/ARG/2008.csv.gz'), compression = 'gzip', index_col = 0)
    print(df.index)

    df = fillinminsdf(df, 15)
    print(df.index)


def get_eventtimes_index(df, eventtimes, missingbetweenerror = True):
    """
    This gets the indexes associated with eventtimes in a df
    Note eventtimes should be sorted
    If a time is not in df, the dataframe returns np.nan

    missingbetweenerror == True: return an error if there are eventtimes that are between df.index[0] and df.index[-1] that are not in df.index
    """
    # need to be sorted for this code to work correctly
    if sorted(eventtimes) != eventtimes:
        raise ValueError("eventtimes should be sorted.")

    if len(eventtimes) == 0:
        return([])

    dfindex = list(df.index)
    lendfindex = len(dfindex)
    # list of missing elements between df.index[0] and df.index[-1]
    missingbetween = []
    eventtimes_index = []
    i = 0
    # go through eventtimes rather than through dfindex to deal better with duplicates
    for eventtime in eventtimes:
        while True:
            # first of all stop if dfindex out of range
            if i >= lendfindex:
                eventtimes_index.append(np.nan)
                break

            # now set as eventtime index if dfindex[i] == eventtime
            # drop if dfindex is above eventtime
            # otherwise move onto next
            if eventtime == dfindex[i]:
                eventtimes_index.append(i)
                break
            elif eventtime < dfindex[i]:
                eventtimes_index.append(np.nan)
                # if i is 0 then eventtime is out of range of dfindex
                if i != 0:
                    # keep record of which eventtimes are between dfindex[0] and dfindex[-1]
                    missingbetween.append(eventtime)
                break
            else:
                # eventtime > dfindex
                # so go to next dfindex and see if matches
                i += 1
                    
    if missingbetweenerror is True:
        if len(missingbetween) > 0:
            raise ValueError("Missing elements in between df.index[0] and df.index[-1]: " + str(missingbetween) + ".")
        

    return(eventtimes_index)
            

def get_eventtimes_index_test():
    df = pd.DataFrame(index = ['20210101_0000M', '20210101_0015M', '20210101_0030M', '20210101_0045M', '20210101_0100M'])
    eventtimes = ['20210101_0015M', '20210101_0045M']

    print('[1, 3] = ' + str( get_eventtimes_index(df, eventtimes) ))


def getdfintrarel_single(df, eventtimes, rpos, ffill = False, bfill = False, ffillfirst = False, eventtimes_index = None):
    """
    df = pandas dataframe of intraday data
    MUST NOT HAVE GAPS since I'm searching by iloc (unlike the daily case)

    eventtimes is the list of minutes that I want
    However, I can also directly specify eventtimes_index to avoid having to get this every time
    eventtimes_index is the indexes of the df corresponding to the relevant events
    I still need eventtimes even if I do specify eventtimes_index so that I can return the df with eventtimes

    rpos is the position relative to eventtimes_index that I want to get

    Note that we do not fill in any missing gaps that don't exist in the data
    """
    if ffillfirst is True:
        if ffill is not False:
            df = df.ffill(limit = ffill)
        if bfill is not False:
            df = df.bfill(limit = bfill)
    else:
        if bfill is not False:
            df = df.bfill(limit = bfill)
        if ffill is not False:
            df = df.ffill(limit = ffill)

    if eventtimes_index is None:
        eventtimes_index = get_eventtimes_index(df, eventtimes, missingbetweenerror = True)

    # adjust by rpos
    eventtimes_index_rpos = [index + rpos for index in eventtimes_index]

    # get which ones are available
    lendf = len(df)
    eventtimes_index_rpos_available_bool = [True if not np.isnan(i) and i >= 0 and i < lendf else False for i in eventtimes_index_rpos]
    eventtimes_available = [eventtimes[i] for i in range(len(eventtimes)) if eventtimes_index_rpos_available_bool[i] is True]
    eventtimes_index_rpos_available = [eventtimes_index_rpos[i] for i in range(len(eventtimes_index_rpos)) if eventtimes_index_rpos_available_bool[i] is True]

    # get df for available times
    # this is the step that takes time
    dfshock = df.iloc[eventtimes_index_rpos_available]

    # save list of eventtimes_relative
    # this list is relative to eventtimes (not to eventtimes_available)
    eventtimes_relative = []
    dfshockindex = list(dfshock.index)
    j = 0
    for i in range(len(eventtimes)):
        if eventtimes_index_rpos_available_bool[i] is True:
            eventtimes_relative.append(dfshockindex[j])
            j += 1
        else:
            eventtimes_relative.append(np.nan)
        
    # set to be original eventtimes
    dfshock.index = eventtimes_available
    # drop any duplicates before reindexing (otherwise won't work)
    dfshock = dfshock.groupby(dfshock.index).first()
    # reindex
    dfshock = dfshock.reindex(eventtimes)

    # Old:{{{
    # # get lists of the times I can actually get from df
    # # also get a list for whether or not I can get a given time from df
    # lendf = len(df)
    # eventtimes_index_rpos_available = []
    # eventtimes_index_rpos_bool = []
    # for i in eventtimes_index_rpos:
    #     if not np.isnan(i) and i >= 0 and i < lendf:
    #         eventtimes_index_rpos_available.append(i)
    #         eventtimes_index_rpos_bool.append(True)
    #     else:
    #         eventtimes_index_rpos_bool.append(False)
    #
    # # get df for available times
    # # this is the step that takes time
    # dfshock = df.iloc[eventtimes_index_rpos_available]
    #
    # I COMMENTED THIS OUT AS CAN JUST DROP MISSING TIMES
    # # add back in times that we weren't able to get from df into the index
    # # this is so the length of dfshock matches the eventtimes I inputted into this function
    # eventtimes_rpos_available = list(dfshock.index)
    # eventtimes_rpos = []
    # for eventtime in eventtimes_index_rpos_bool:
    #     if eventtime is True:
    #         eventtimes_rpos.append(eventtimes_rpos_available.pop(0))
    #     else:
    #         # give a clearly bad time
    #         eventtimes_rpos.append("dropped")
    #
    # # adjust index so covers all periods
    # dfshock = dfshock.reindex(eventtimes_rpos)
    # Old:}}}

    return(dfshock, eventtimes_relative)


def getdfintrarel_single_test():
    # load data
    df = pd.read_csv(macrodataexternal / Path('int-stock-intra/output/merge/USA.csv.gz'), compression = 'gzip', index_col = 0)

    # fill in minutes
    df = fillinminsdf(df, 5)

    eventtimes = ['19950101_0000M', '19960102_0830M', '20100506_1845M']

    # get event indexes
    # first time outside of data
    # second time first point of data (so rel=-1 should yield NaN)
    # thid time: considering flash crash of 2:45 US Eastern Time i.e. the flash crash on May 6 2010
    # started at 2:32pm and lasted about 36 minutes
    eventtimes_index = get_eventtimes_index(df, eventtimes)

    # the first two events should be NaN
    dfshock, eventtimes_relative = getdfintrarel_single(df, eventtimes, -1, ffill = 2, eventtimes_index = eventtimes_index)
    print(dfshock)
    print(eventtimes_relative)


def getdfintrarel_reldict(dfclose, eventtimes, reldict, yearlist = None, dfopen = True):
    """
    Idea of function is to get many getdfintrarel using a reldict
    Each singledict in reldict contains similar arguments to running getdfintrarel_single once

    dfclose = pandas dataframe of intraday data with indexes of format %y%m%d_%H%MM
    Exception:
    If yearlist is specified then dfclose is a list of pandas DataFrames for each year in yearlist in with the minute indexes

    If doing bfill/ffill: DATA MUST BE FILLED IN so covers every X minutes (see fillinminsdf)
    If don't want gaps and events are not every X minutes: DATA MUST BE ROUNDED (see roundnearestinterval)

    eventtimes - times of events

    reldict = {'m1h': {'rpos': -4, 'bfill': False, 'ffill': False, 'open': True}}

    If open is True:
    dfopen must be defined
    dfopen must have the same index as dfclose
    dfopen must be filled in if bfill/ffill are used
    """

    # get eventtimes_list and dflist
    if yearlist is not None:
        # verify no issues with years
        if sorted(yearlist) != yearlist:
            print(yearlist)
            raise ValueError('yearlist not in order.')
        for year in yearlist:
            if len(str(year)) != 4 or isinstance(year, int) is False:
                raise ValueError('year in df yeardict is misspecified.')
        if len(yearlist) != len(set(yearlist)):
            print(yearlist)
            raise ValueError('Duplicates in df yeardict.')

        # list of events by available year
        eventtimes_list = []
        eventtime_i = 0
        for year in yearlist:
            eventtimes_thisyear = []
            # add events in this way so that events in years that are not covered by the data are still included in the next year
            while True:
                # no more events to consider in this case
                if eventtime_i == len(eventtimes):
                    break
                # add any events to this year that came in this year or prior
                if eventtimes[eventtime_i] < str(year + 1):
                    eventtimes_thisyear.append(eventtimes[eventtime_i])
                    eventtime_i = eventtime_i + 1
                else:
                    break
            eventtimes_list.append(eventtimes_thisyear)
        # add remaining events that have not been added to the last year of eventtimes_list
        # this is needed if eventtimes occur after last year of yearlist
        if eventtime_i == 0:
            eventtimes_list = [eventtimes]
        else:
            eventtimes_list[-1] = eventtimes_list[-1] + eventtimes[eventtime_i: ]

        dfcloselist = dfclose
        if dfopen is not False:
            dfopenlist = dfopen
    else:
        eventtimes_list = [eventtimes]
        dfcloselist = [dfclose]
        if dfopen is not False:
            dfopenlist = [dfopen]

    # get eventtimes_index_list
    eventtimes_index_list = []
    for i in range(len(dfcloselist)):
        eventtimes_index_list.append( get_eventtimes_index(dfcloselist[i], eventtimes_list[i]) )

    for relname in reldict:
        singledict = reldict[relname]

        if 'rpos' not in singledict:
            raise ValueError('rpos not defined for relname: ' + relname + '.')
        if 'bfill' not in singledict:
            singledict['bfill'] = False
        if 'ffill' not in singledict:
            singledict['ffill'] = False
        if 'ffillfirst' not in singledict:
            singledict['ffillfirst'] = False
        if 'open' not in singledict:
            singledict['open'] = False

        if singledict['open'] is True and dfopen is None:
            raise ValueError('open is True so need to specify a dfopen.')

        if singledict['open'] is True:
            dfthislist = dfopenlist
        else:
            dfthislist = dfcloselist

        dfrellist = []
        eventtimes_relative = []
        for i in range(len(dfcloselist)):
            dfrel_i, eventtimes_relative_i = getdfintrarel_single(dfthislist[i], eventtimes_list[i], singledict['rpos'], ffill = singledict['ffill'], bfill = singledict['bfill'], ffillfirst = singledict['ffillfirst'], eventtimes_index = eventtimes_index_list[i])

            dfrellist.append(dfrel_i)
            eventtimes_relative = eventtimes_relative + eventtimes_relative_i

        dfrel = pd.concat(dfrellist)
        reldict[relname]['relativedates'] = eventtimes_relative
        reldict[relname]['dfrel'] = dfrel

    return(reldict)


def getdfintrarel_reldict_test():

    # get data:{{{
    # load data
    df = pd.read_csv(macrodataexternal / Path('int-stock-intra/output/merge/USA.csv.gz'), compression = 'gzip', index_col = 0)

    # fill in minutes
    df = fillinminsdf(df, 15)

    # separate df into open and closed
    openvars = [col for col in df.columns if col.split('__')[2].endswith('_op')]
    dfclose = df[[col for col in df.columns if col not in openvars]]
    dfopen = df[openvars]
    # rename dfopen
    dfopen.columns = [col.replace('_op__', '__') for col in dfopen.columns]
    # get data:}}}

    # get event times:{{{
    eventtimes = ['20100506_1845M']
    # get event times:}}}

    # this will take the last available point in the period 20100506_1645M - 20100506_1745M
    # and the first available point in the period 20100506_1945M - 20100506_2045M
    reldict = {'m1h': {'rpos': -5, 'ffill': 3}, '1h': {'rpos': 4, 'bfill': 3, 'open': True}}

    reldict = getdfintrarel_reldict(dfclose, eventtimes, reldict, dfopen = dfopen)

    print(reldict['m1h']['dfrel'])
    print(reldict['1h']['dfrel'])


# Runall:{{{1
def testall():
    getrelativedates_test()
    getdfdailyrel_single_test1()
    getdfdailyrel_single_test2()
    getdfdailyrel_reldict_test(printdetails = True)

    fillinminsdf_test()
    get_eventtimes_index_test()
    getdfintrarel_single_test()
    getdfintrarel_reldict_test()
