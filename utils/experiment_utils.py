import os.path as path
from contextlib import redirect_stdout

from model.flags import FLAGS


def redirects_stdout(func):
    def capture_wrapper(*args, **kwargs):
        if FLAGS.redirect_stdout:
            with open(path.join(path.dirname(FLAGS.output_dir), FLAGS.output_pattern + '.log'), 'a') as f:
                with redirect_stdout(f):
                    results = func(*args, **kwargs)
                    f.flush()
        else:
            results = func(*args, **kwargs)

        return results

    return capture_wrapper


def print_config():
    print("================= Configuration =================")
    for flag, value in FLAGS.flag_values_dict().items():
        print(flag + ": " + str(value))
    print("\n")
