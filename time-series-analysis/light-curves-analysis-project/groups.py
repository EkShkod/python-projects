import numpy as np
from matplotlib import pyplot as plt


class Blockject():
    """ A single Bayesian Block object with edge index, edge times, mean flux and mean error properties """
    def __init__(self, edges, indx, flux, error):
        self.start_time, self.end_time = edges
        self.start_indx, self.end_indx = indx
        self.flux = flux
        self.error = error
        self.dur = self.end_time - self.start_time


def crea_blockjects(block_val, block_val_error, edge_index, edges):
    """ Creates a Beaysian Block objects list """
    blocks = []
    for i,_ in enumerate(block_val):
        edgs = (edges[i], edges[i+1])
        indx = (edge_index[i], edge_index[i+1])
        flux = block_val[i]
        error = block_val_error[i]
        block = Blockject(edgs, indx, flux, error)
        blocks.append(block)
    return blocks


def find_block_groups(blocks, m, flux, band):
    """ Find groups of Bayesian Blocks satisfying the conditions from Meyer et al., 2019 """
    # Step 1: find all peak blocks above F_max
    peak_blocks = []
    if band == 'optical':
        F_max = m
        F_min = np.mean(flux)
        for block in blocks:
            if block.flux <= F_max:
                peak_blocks.append(block)

    elif band == 'gamma':
        F_max = m * np.mean(flux)
        F_min = np.mean(flux)
        for block in blocks:
            if block.flux >= F_max:
                peak_blocks.append(block)


    # Step 2: find supplemental blocks above F_min
    suppl_blocks = []
    for p_block in peak_blocks:
        i = list(blocks).index(p_block)
        if i != 0 and i != len(blocks):
            l = i - 1
            r = i + 1
            if band == 'optical':
                while blocks[l].flux <= F_min and l >= 0:
                    suppl_blocks.append(blocks[l])
                    l -= 1
                while r < len(blocks) and blocks[r].flux <= F_min:
                    suppl_blocks.append(blocks[r])
                    r += 1

            if band == 'gamma':
                while blocks[l].flux >= F_min and l >= 0:
                    suppl_blocks.append(blocks[l])
                    l -= 1
                while r < len(blocks) and blocks[r].flux >= F_min:
                    suppl_blocks.append(blocks[r])
                    r += 1

    # Step 3: merge blocks and add right and left blocks
    all_peaks = np.hstack([peak_blocks, suppl_blocks])
    all_peaks = list(set(all_peaks))
    all_peaks.sort(key=lambda x: x.start_time)
    end_blocks = []
    for block in all_peaks:
        i = list(blocks).index(block)
        if i != 0 and i != len(blocks)-1:
            end_blocks.append(blocks[i-1])
            end_blocks.append(blocks[i+1])
    bb_groups = np.hstack([all_peaks, end_blocks])
    bb_groups = list(set(bb_groups))
    bb_groups.sort(key=lambda x: x.start_time)
    return bb_groups


def plot_bb_groups(bb_groups, lc, ax=None):
    if ax is None:
        ax = plt.gca()
    for block in bb_groups:
        y = np.max(lc.flux)
        ax.hlines(y, block.start_time, block.end_time, color='black', lw=3)
    return
