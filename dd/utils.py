import numpy as np

def boundary_points(n_steps, n_regions):
    """Location of boundary interfaces if overlap=0
    """
    k = int(n_steps/n_regions)
    points = [k*i for i in range(1, n_regions)]
    points = [1] + points + [n_steps]
    return [p-1 for p in points]
    #return np.array(points, np.int) - 1

def region_slice_index(n_steps, n_regions, overlap):
    """Slice indicies of regions in an array.
    """    
    min_point = 0
    max_point = n_steps-1
    bp = boundary_points(n_steps, n_regions)
            
    rsi = []
    for i in range(n_regions):
        start = bp[i]
        stop  = bp[i+1]
        if start > min_point:
            start -= overlap
        if stop  < max_point:
            stop  += overlap
        rsi.append((start, stop+1))
    return rsi

def region_midpoints(n_steps, n_regions, overlap):
    """The approximate middle of each region.
    """
    midpoints = []
    for rsi in region_slice_index(n_steps, n_regions, overlap):
        midpoints.append(int(rsi[1]+rsi[0])/2)
        
    return midpoints

def region_views(arr, n_regions, overlap):
    """Views of all regions in an array.
    """    
    n_steps = len(arr)  
    rsi = region_slice_index(n_steps, n_regions, overlap) 
    
    views = []
    for idx in rsi:
        views.append(arr[idx[0]:idx[1]])
    return views

def region_avg_denom(n_steps, n_regions, overlap):
    """An array of the number of overlap counts.
    """
    base = np.zeros(n_steps, dtype=np.float)
    for rv in region_views(base, n_regions, overlap):
        rv += 1
    return base
