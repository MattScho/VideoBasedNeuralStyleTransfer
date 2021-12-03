'''
Read in configuration and run application

:author: Matthew Schofield
'''
from managers.configuration_manager import Configuration
from argparse import ArgumentParser
from main import main
import os

# Get current directory path
absolute_path = os.path.dirname(os.path.abspath(__file__))

# Parse configuration file argument
parser = ArgumentParser()
parser.add_argument("-c", help="Configuration file to pass")
args = parser.parse_args()

configuration = Configuration(args.c, absolute_path)

# Run main application with configuration set-up
main(configuration)
