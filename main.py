import logging

import my
from my import *
from collections import Counter

generator = Generator1(logging.getLogger('generator'))
generator.start()
reader = generator.reader