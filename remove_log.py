import os

import settings


try:
    os.system('rm -r {}'.format(settings.DIR_logs))
except Exception as e:
    print(e)

try:
    os.system('rm -r {}'.format(settings.DIR_tblogs))
except Exception as e:
    print(e)

try:
    os.system('rm -r {}'.format(settings.DIR_trained_model))
except Exception as e:
    print(e)

try:
    os.system('rm -r {}'.format(settings.DIR_confusions))
except Exception as e:
    print(e)