'''

Implementation of 

https://gfx.cs.princeton.edu/pubs/Weyrich_2007_DBF/relief.pdf

Run this as:

python ./main.py -i slc_depth.png -i slc_bas_relief.png

'''

import argparse

import numpy as np
import scipy.sparse.linalg
import tqdm

import pyoptim.helpers as H

# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gc_alpha', type=float, default=10.0, help='Gradient compression alpha factor')
    parser.add_argument('--ofs', type=int, default=0, help='Gradient offset, set to 1 to use two-sided difference')    

    parser.add_argument('-i', '--inputfile', type=str, help='Input depth image' )
    parser.add_argument('-o', '--outputfile', type=str, help='Output bas relief image' )

    args = parser.parse_args()
    
    if not args.inputfile:
        parser.error('Please specify an input file with the -i or --inputfile option')

    if not args.outputfile:
        parser.error('Please specify an output file with the -o or --outputfile option')
    
    return args

# ---------------------------

def grad_compress( gx, gy, alpha=10.0 ):
    mags = np.sqrt( gx*gx + gy*gy + 0.0001 )
    cmags = (1.0/alpha) * np.log( 1.0 + alpha*mags )
    gx = gx / cmags
    gy = gy / cmags
    return gx, gy, cmags

def rc_to_linind( r,c ):
    return r*COLS+c

# ---------------------------

# args = parse_args()
INFILE = "tmp/composite_output.png"
OUTFILE = "rafael.png"
WIDTH = HEIGHT = 256
OFS = 0
GC_ALPHA = 10

# ---------------------------

# load depth image
hf = H.read_image(INFILE, w=WIDTH, h=HEIGHT, format="L")
print(np.min(hf), np.max(hf))

# we expect depth to increase with distance.  Flip it around.
# hf = (255.0-hf)/255.0

bkg_h = np.min( hf )

ROWS, COLS = hf.shape

# boundaries
hf[0,:]=0.0
hf[ROWS-1,:]=0.0
hf[:,0]=0.0
hf[:,COLS-1]=0.0

# ---------------------------

#
# calc gx,gy - XXX I know I should be using a convolution for this.  :(
#

print("Calculating image gradient")
gx = np.zeros((ROWS,COLS),dtype=np.float32)
gy = np.zeros((ROWS,COLS),dtype=np.float32)
for row_ind in tqdm.trange(1,ROWS-1):
    for col_ind in range(1,COLS-1):
        gx[row_ind,col_ind] = hf[row_ind,col_ind+OFS]-hf[row_ind,col_ind-1]
        gy[row_ind,col_ind] = hf[row_ind+OFS,col_ind]-hf[row_ind-1,col_ind]        

# compress gradients
print("Compressing gradients")
comp_gx, comp_gy, cmags = grad_compress( gx, gy, GC_ALPHA)

#
# construct sparse matrix representing ls equation
#

print("Constructing dx constraints")

row_inds = []
col_inds = []
data = []
b = []
next_ind = ROWS*COLS
for row_ind in tqdm.trange(ROWS):
    for col_ind in range(COLS):
        xlc   = rc_to_linind( row_ind, col_ind )
        # boundary condition
        if row_ind == 0 or row_ind == ROWS-1 or col_ind == 0 or col_ind == COLS-1: # or hf[row_ind,col_ind] == bkg_h:
            row_inds.append(xlc); col_inds.append(xlc); data.append(1);
            b.append( 0 )
            continue
        
        # dx constraint
        xlc_l = rc_to_linind( row_ind, col_ind-1 )
        xlc_r = rc_to_linind( row_ind, col_ind+OFS )
        row_inds.append(xlc); col_inds.append(xlc_l); data.append(-1);
        row_inds.append(xlc); col_inds.append(xlc_r); data.append(1);
        # target value
        b.append( comp_gx[row_ind,col_ind] )

print("Constructing dy constraints")

next_ind = ROWS*COLS
for row_ind in tqdm.trange(ROWS):
    for col_ind in range(COLS):
        xlc   = rc_to_linind( row_ind, col_ind )
        # boundary condition
        if row_ind == 0 or row_ind == ROWS-1 or col_ind == 0 or col_ind == COLS-1: # or hf[row_ind,col_ind] == bkg_h:
            # already got these above
            continue
        
        # dy constraint
        xlc_u = rc_to_linind( row_ind-1, col_ind )
        xlc_d = rc_to_linind( row_ind+OFS, col_ind )
        row_inds.append(next_ind); col_inds.append(xlc_u); data.append(-1);  
        row_inds.append(next_ind); col_inds.append(xlc_d); data.append(1);
        # target value
        b.append( comp_gy[row_ind,col_ind] )
        
        next_ind += 1

print("Constructing sparse matrix")
A = scipy.sparse.csc_matrix( (data,(row_inds,col_inds)) )
b = np.atleast_2d( b ).T

#
# Solve the least-squares reconstruction problem
#

print("Solving!")

results = scipy.sparse.linalg.lsqr( A, b, show=True, atol=1e-15, btol=1e-15, conlim=100 )

new_hf = np.reshape( np.atleast_2d( results[0] ).T, (ROWS,COLS) )

new_hf_img = new_hf - np.min(new_hf)
new_hf_img = new_hf_img / np.max(new_hf_img)

import matplotlib.pyplot as plt
plt.imsave(OUTFILE, new_hf_img, cmap='gray')
print(np.min(new_hf_img), np.max(new_hf_img))