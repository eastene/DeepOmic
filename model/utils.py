from model.flags import FLAGS
from io import StringIO

import os.path as path
from contextlib import redirect_stdout


def redirects_stdout(func):
    def capture_wrapper(*args, **kwargs):
        if FLAGS.redirect_stdout:
            with open(path.join(path.dirname(FLAGS.output_dir), FLAGS.output_pattern + '.log'), 'a') as f:
                with redirect_stdout(f):
                    results = func(*args, **kwargs)
        else:
            results = func(*args, **kwargs)

        return results

    return capture_wrapper
