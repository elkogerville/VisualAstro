'''
Author: Elko Gerville-Reache
Date Created: 2025-12-10
Date Modified: 2025-12-10
Description:
    Utility functions for DataCube and FitsFile.
Dependencies:
    - astropy
    - numpy
'''

from astropy.time import Time

def log_history(header, message):
    '''
    Add `HISTORY` entry to header.

    Parameters
    ––––––––––
    header : astropy.Header
    message : str
    '''
    timestamp = Time.now().isot
    log = f'{timestamp} {message}'

    header.add_history(log)
