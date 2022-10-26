########################################################
# Redirect benchmark report to out/fileName.txt
########################################################

def write_benchmark_report(print_fun, path, mode='w', globalVars=None, localVars=None, msg=None):
    """
    Parameters
    ----------
    print_func (str): script to print output\n
    path (str): path to store the `.txt` output file\n
    mode (str): 'w' or 'a'
    mgs (str): message to print when finished
    """
    import sys
    original_stdout = sys.stdout
    with open(path, mode) as f: 
        sys.stdout = f
        exec(print_fun, globalVars, localVars)
        sys.stdout = original_stdout # Reset the standard output to its original value
    if msg is not None: print(f'{msg} saved to file {path}')