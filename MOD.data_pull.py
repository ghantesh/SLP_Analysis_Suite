# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:37:46 2018

@author: amaan
Purpose: Data Pull Module of the SLP analysis suite. The module houses two functions. The 
functions pull data from the ltx tree. For the module to work the host computer needs to be 
the pppl network, either through vpn or directly installed in one of the ltx computers. 

1. get_singleprobedata(shot_no,tree_loc,tree_name) - takes shot number, tree location eg 
'lithos' and tree name eg, 'ltx'/'ltx_b'. Inputs 2 and 3 are strings, input one is an integer.
The function should out V,I,t for that particular shot.

2. get_shotparams(shot_no,tree_loc,tree_name) - has the same inputs as 1, should out, Ip, ne, 
ECH and v_loop.
"""


