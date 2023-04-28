#!/bin/bash

# Print usage instructions
function print_usage {
  echo "Usage: start_flask_server.sh [-e environment] [-d 0|1]"
  echo ""
  echo "  -e environment  Set the Flask environment to development or production (default: development)"
  echo "  -d 0|1          Enable/disable Flask debug mode (default: 0)"
  echo ""
  echo "Examples:"
  echo "  ./start_flask_server.sh                # Start the server with default settings"
  echo "  ./start_flask_server.sh -e production  # Start the server in production mode"
  echo "  ./start_flask_server.sh -d 1           # Start the server with debug mode enabled"
  echo "  ./start_flask_server.sh -e production -d 1  # Start the server in production mode with debug mode enabled"
}

export FLASK_APP=main.py
# Parse command-line arguments
while getopts ":e:d:" opt; do
  case ${opt} in
    e ) # Set the environment
      export FLASK_ENV=${OPTARG}
      ;;
    d ) # Enable/disable debug mode
      export FLASK_DEBUG=${OPTARG}
      ;;
    \? ) # Invalid option
      echo "Error: Invalid option -$OPTARG"
      echo ""
      print_usage
      exit 1
      ;;
    : ) # Missing argument
      echo "Error: Option -$OPTARG requires an argument."
      echo ""
      print_usage
      exit 1
      ;;
  esac
done

# Start the Flask server
python3 -m flask run