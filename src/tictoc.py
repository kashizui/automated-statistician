import inspect
import timeit


start = None
label = None


def tic(set_label=None):
    """
    Use in combination with toc() to do some quick profiling.

    ex.
        tic("hello")
        ... some code ...
        toc()

        => "hello: x.xxxxx seconds"
    """
    global start
    global label
    start = timeit.default_timer()
    label = set_label


def toc():
    elapsed = timeit.default_timer() - start
    (frame, filename, line_number,
     function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[1]
    if label is None:
        print "%s:%d: %f seconds" % (filename, line_number, elapsed)
    else:
        print "%s: %f seconds" % (label, elapsed)
    return elapsed
