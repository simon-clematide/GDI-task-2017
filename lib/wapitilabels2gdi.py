#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import sys
from optparse import OptionParser

"""
Module for XXX

"""

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())


def process(options=None, args=None):
    """Do the processing"""
    if options.debug:
        print(options, file=sys.stderr)
        print(args, file=sys.stderr)
    tag = None
    for l in sys.stdin:
        l = l.strip()
        if l == "" and tag is not None:
            print(tag)
            tag = None
        else:
            tag = l
    if tag is not None:
        print(tag)


def main():
    """
    Invoke this module as a script
    """

    parser = OptionParser(
        usage='%prog [OPTIONS] [ARGS...]',
        version='%prog 0.99',  #
        description='Calculate something',
        epilog='Contact simon.clematide@uzh.ch'
    )
    parser.add_option('-l', '--logfile', dest='logfilename',
                      help='write log to FILE', metavar='FILE')
    parser.add_option('-q', '--quiet',
                      action='store_true', dest='quiet', default=False,
                      help='do not print status messages to stderr')
    parser.add_option('-d', '--debug',
                      action='store_true', dest='debug', default=False,
                      help='print debug information')

    (options, args) = parser.parse_args()

    process(options=options, args=args)


if __name__ == '__main__':
    main()
