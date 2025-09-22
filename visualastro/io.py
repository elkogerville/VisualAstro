import matplotlib.pyplot as plt

def save_figure_2_disk(dpi=600):
    '''
    Saves current figure to disk as a pdf, png, or svg,
    and prompts user for a filename and format.
    Parameters
    ––––––––––
    dpi: float
        Resolution in dots per inch.
    '''
    allowed_formats = {'pdf', 'png', 'svg'}
    # prompt user for filename, and extract extension
    filename = input('Input filename for image (ex: myimage.pdf): ').strip()
    basename, *extension = filename.rsplit('.', 1)
    # if extension exists, and is allowed, extract extension from list
    if extension and extension[0].lower() in allowed_formats:
        extension = extension[0]
    # else prompt user to input a valid extension
    else:
        extension = ''
        while extension not in allowed_formats:
            extension = input(f'Please choose a format from ({", ".join(allowed_formats)}): ').strip().lower()
    # construct complete filename
    filename = f'{basename}.{extension}'

    # save figure
    plt.savefig(filename, format=extension, bbox_inches='tight', dpi=dpi)
