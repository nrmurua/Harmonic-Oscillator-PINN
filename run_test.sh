export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -m unittest discover -s test -p "test_*.py"