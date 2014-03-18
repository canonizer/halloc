#!/usr/bin/env python
import numpy as np #numerical stuff
import sys
import os

import prettyplotlib as ppl # makes nicer colors and generally better to look at graphs
import matplotlib.pyplot as plt
import matplotlib as mpl
from prettyplotlib import brewer2mpl

def funlink(path):
  try:
    os.unlink(path)
  except:
    pass

inputFileName = "exp-log-thru.csv"
data = np.loadtxt(inputFileName, skiprows=1)

# filtering the numpy array for specific sz and l values
def np_filter(data, l, sz):
  nps = data.shape[0]
  return np.array([data[i,:] for i in range(nps) if
              data[i,0]==sz and data[i,1]==l and data[i,2]>=1024])

for l in [1, 4]:
  for sz in [16, 256]:
    if(sz == 256 and l == 4):
      continue
    curData = np_filter(data, l, sz)
    xs = range(curData.shape[0])
    
    # allocation throughput for different sizes
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ppl.plot(ax, xs, curData[:,3], '-o', label="Private", linewidth=2)
    ppl.plot(ax, xs, curData[:,4], '-o', label="Spree", linewidth=2)
    ppl.plot(ax, xs, curData[:,5], '-o', label="Spree malloc", linewidth=2)
    ppl.plot(ax, xs, curData[:,6], '-o', label="Spree free", linewidth=2)
    ax.set_title("%dx%d B Throughput" % (l,sz));
    ax.set_xlabel("#threads, x 1024")
    ax.set_ylabel("Throughput, Mops/s")
    ax.set_xticks(xs)
    ax.set_xticklabels(['%.0lf' % d for d in curData[:,2] / 1024])
    ax.axis(xmin=-1, xmax=len(xs), ymin=0)
    ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)
    ppl.legend(ax, loc=0)
    plt.tick_params(axis='both', which='major', direction='in', bottom=True)
    outputfilename = '%dx%db-thru.pdf' % (l,sz)
    funlink(outputfilename)
    fig.savefig(outputfilename, dpi=300, bbox_inches='tight')


inputFileName = "exp-log-lat.csv"
data = np.loadtxt(inputFileName, skiprows=1)
l = 1
for sz in [16, 256]:
  for iaction in [0, 1]:
    actions = ['Malloc', 'Free']
    action = actions[iaction]
    curData = np_filter(data, l, sz)
    xs = range(curData.shape[0])  
    # allocation throughput for different sizes
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    divd=0.732*1000
    ppl.plot(ax, xs, curData[:,3+3*iaction]/divd, '-o', label="Min", linewidth=2)
    ppl.plot(ax, xs, curData[:,4+3*iaction]/divd, '-o', label="Avg", linewidth=2)
    ppl.plot(ax, xs, curData[:,5+3*iaction]/divd, '-o', label="Max", linewidth=2)
    ax.set_title("%s Latency" % (action));
    ax.set_xlabel("#threads, x 1024")
    ax.set_ylabel("Latency, us")
    ax.set_xticks(xs)
    ax.set_xticklabels(['%.0lf' % d for d in curData[:,2] / 1024])
    ax.set_yscale('log')
    ax.axis(xmin=-1, xmax=len(xs), ymin=1)
    ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)
    ppl.legend(ax, loc=0)
    plt.tick_params(axis='both', which='major', direction='in', bottom=True)
    outputfilename = '1x%dB-lat-%s.pdf' % (sz,action.lower())
    funlink(outputfilename)
    fig.savefig(outputfilename, dpi=300, bbox_inches='tight')
