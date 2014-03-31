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

# filtering the numpy array for specific sz and l values
def np_filter(data, sz):
  nps = data.shape[0]
  return np.array([data[i,:] for i in range(nps) if
                   data[i,0]==sz])

inputFileName = "exp-log-priv.csv"
data = np.loadtxt(inputFileName, skiprows=1, usecols=[1,2,3,4])

allocators = ['Halloc', 'ScatterAlloc', 'CUDA']
#allocators = ['Halloc']

nps0 = data.shape[0]
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.set_yscale('log')
#ymin = 1
ymin = np.amin(data[:,3]) / 1.5
ymax = np.amax(data[:,3]) * 1.5
#ymin = 0.1
#ymax = 5 * 10**3
for ialloc in range(len(allocators)):
  for sz in [16, 64]:
    l = 1
    if(sz == 16):
      l = 4
    alloc = allocators[ialloc];
    curData = data[np.array(range(nps0/3))*3 + ialloc, :]
    curData = np_filter(curData, sz)
    xs = range(curData.shape[0])
    # allocation throughput for different sizes
    ppl.plot(ax, xs, curData[:,3], '-o',
             label=('%dx%d B %s' % (l,sz,alloc)), linewidth=2)
    ax.set_xlabel('#threads, x 1024')
    ax.set_ylabel('Throughput, Mops/s')
    if(ialloc == len(allocators) - 1 and sz == 64):
      ax.set_xticks(xs)
      ax.set_xticklabels(['%.0lf' % d for d in curData[:,2] / 1024.0])
      ax.axis(xmin=-1, xmax=len(xs), ymin=ymin, ymax=ymax)
      ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)

ax.set_title('Private Test Pair Throughput')
ppl.legend(ax, loc=0)
plt.tick_params(axis='both', which='major', direction='in', bottom=True)
outputfilename = 'vs-priv-pair.pdf'
funlink(outputfilename)
fig.savefig(outputfilename, dpi=300, bbox_inches='tight')
  
#plt.show()


inputFileName = "exp-log-spree.csv"
data = np.loadtxt(inputFileName, skiprows=1, usecols=[1,2,3,4,5,6])

allocators = ['Halloc', 'ScatterAlloc', 'CUDA']
nps0 = data.shape[0]
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.set_yscale('log')
ymin = np.amin(data[:,4]) / 1.5
ymax = np.amax(data[:,4]) * 1.5
for ialloc in range(len(allocators)):
  for sz in [16, 64]:
    l = 1
    if(sz == 16):
      l = 4
    alloc = allocators[ialloc];
    curData = data[np.array(range(nps0/3))*3 + ialloc, :]
    curData = np_filter(curData, sz)
    xs = range(curData.shape[0])
    # allocation throughput for different sizes
    ppl.plot(ax, xs, curData[:,4], '-o',
             label=('%dx%d B %s' % (l,sz,alloc)), linewidth=2)
    ax.set_xlabel('#threads, x 1024')
    ax.set_ylabel('Throughput, Mops/s')
    if(ialloc == len(allocators) - 1 and sz == 64):
      ax.set_xticks(xs)
      ax.set_xticklabels(['%.0lf' % d for d in curData[:,2] / 1024.0])
      ax.axis(xmin=-1, xmax=len(xs), ymin=ymin, ymax=ymax)
      ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)

ax.set_title('Spree Test malloc() Throughput')
ppl.legend(ax, loc=0)
plt.tick_params(axis='both', which='major', direction='in', bottom=True)
outputfilename = 'vs-spree-malloc.pdf'
funlink(outputfilename)
fig.savefig(outputfilename, dpi=300, bbox_inches='tight')
  
#plt.show()
