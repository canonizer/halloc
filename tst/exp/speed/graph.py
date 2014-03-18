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

# change font to Open Sans (has some kerning issues, though)
#mpl.rcParams.update({'font.family':'Open Sans'})

inputFileName = "exp-log-single.csv"

data = np.loadtxt(inputFileName, skiprows=1)
xs = range(data.shape[0])

# allocation speed for different sizes
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ppl.plot(ax, xs, data[:,1], '-o', label="Private", linewidth=2)
ppl.plot(ax, xs, data[:,3], '-o', label="Spree", linewidth=2)
ppl.plot(ax, xs, data[:,4], '-o', label="Spree malloc", linewidth=2)
ax.set_title("Allocation Speed for Different Sizes");
ax.set_xlabel("Allocation size, B")
ax.set_ylabel("Speed, GiB/s")
ax.set_xticks(xs)
ax.set_xticklabels(['%.0lf' % d for d in data[:, 0]])
ax.axis(xmin=-1, ymin=0)
ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)
ppl.legend(ax, loc=0)
plt.tick_params(axis='both', which='major', direction='in', bottom=True)
outputfilename = 'single-speed.pdf'
funlink(outputfilename)
fig.savefig(outputfilename, dpi=300, bbox_inches='tight')

# allocation throughput for different sizes
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ppl.plot(ax, xs, data[:,2], '-o', label="Private", linewidth=2)
ppl.plot(ax, xs, data[:,5], '-o', label="Spree", linewidth=2)
ppl.plot(ax, xs, data[:,6], '-o', label="Spree malloc", linewidth=2)
ax.set_title("Allocation Throughput for Different Sizes");
ax.set_xlabel("Allocation size, B")
ax.set_ylabel("Throughput, Mop/s")
ax.set_yscale('log')
ax.set_xticks(xs)
ax.set_xticklabels(['%.0lf' % d for d in data[:, 0]])
ax.axis(xmin=-1, ymin=1)
ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)
ppl.legend(ax, loc=0)
plt.tick_params(axis='both', which='major', direction='in', bottom=True)
outputfilename = 'single-thru.pdf'
funlink(outputfilename)
fig.savefig(outputfilename, dpi=300, bbox_inches='tight')

# size combinations
data = np.loadtxt('exp-log-combi.csv', skiprows=1, usecols=[1, 2, 3, 4, 5, 6])
labels = ['8..32', '8..64', '8..256', '8..3072']
xs = np.array(range(data.shape[0])) * 2
step=0.3
width=0.25

# allocation speed for different size combinations
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ppl.bar(ax, xs, data[:,0], color='b', width=width, label="Private")
ppl.bar(ax, xs + step, data[:,2], color='g', width=width, label="Spree")
ppl.bar(ax, xs + 2*step, data[:,3], color='r', width=width, label="Spree malloc")
ax.set_title("Allocation Speed for Combinations of Sizes");
ax.set_xlabel("Allocation size, B")
ax.set_ylabel("Speed, GiB/s")
ax.set_xticks(xs + 0.45)
ax.set_xticklabels(labels)
ax.axis(xmin=-0.5, ymin=1)
ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)
ppl.legend(ax, loc=0)
outputfilename = 'combi-speed.pdf'
funlink(outputfilename)
fig.savefig(outputfilename, dpi=300, bbox_inches='tight')

# allocation throughput for different size combinations
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ppl.bar(ax, xs, data[:,1], color='b', width=width, label="Private")
ppl.bar(ax, xs + step, data[:,4], color='g', width=width, label="Spree")
ppl.bar(ax, xs + 2*step, data[:,5], color='r', width=width, label="Spree malloc")
ax.set_title("Allocation Throughput for Combinations of Sizes");
ax.set_xlabel("Allocation size, B")
ax.set_ylabel("Throughput, Mop/s")
ax.set_xticks(xs + 0.45)
ax.set_xticklabels(labels)
ax.axis(xmin=-0.5, ymin=1)
ax.grid(axis='y', color='0.3', linestyle=':', antialiased=True)
ppl.legend(ax, loc=0)
outputfilename = 'combi-thru.pdf'
funlink(outputfilename)
fig.savefig(outputfilename, dpi=300, bbox_inches='tight')

#plt.show()
