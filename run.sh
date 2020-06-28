#!/usr/bin/env bash
set -euo pipefail
pip install pyfiglet
clear
pyfiglet -f slant "Welcome to"
pyfiglet -f slant "EEE415 Lab3"

mkdir -p fig
mkdir -p model
mkdir -p data

# baseline implement
python d-dash_hc.py -N 'FNN' -R 300 -T False -F 'FNN_without_target_net_B500' -B 500
python d-dash_hc.py -N 'LSTM' -R 300 -T False -F 'LSTM_without_target_net_B500' -B 500
python d-dash_hc.py -N 'FNN' -R 300 -T True -F 'FNN_with_target_net_B500' -B 500
python d-dash_hc.py -N 'LSTM' -R 300 -T True -F 'LSTM_with_target_net_B500' -B 500

# change batch size
# change the batch size to 100
python d-dash_hc.py -N 'FNN' -R 300 -T False -F 'FNN_without_target_net_B100' -B 100
python d-dash_hc.py -N 'LSTM' -R 300 -T False -F 'LSTM_without_target_net_B100' -B 100
python d-dash_hc.py -N 'FNN' -R 300 -T True -F 'FNN_with_target_net_B100' -B 100
python d-dash_hc.py -N 'LSTM' -R 300 -T True -F 'LSTM_with_target_net_B100' -B 100

# change the batch size to 1000
python d-dash_hc.py -N 'FNN' -R 300 -T False -F 'FNN_without_target_net_B1000' -B 1000
python d-dash_hc.py -N 'LSTM' -R 300 -T False -F 'LSTM_without_target_net_B1000' -B 1000
python d-dash_hc.py -N 'FNN' -R 300 -T True -F 'FNN_with_target_net_B1000' -B 1000
python d-dash_hc.py -N 'LSTM' -R 300 -T True -F 'LSTM_with_target_net_B1000' -B 1000

# change channel history
# change the channel history to 1
python d-dash_hc.py -N 'FNN' -R 300 -T False -F 'FNN_without_target_net_C1' -B 500 -H 1
python d-dash_hc.py -N 'LSTM' -R 300 -T False -F 'LSTM_without_target_net_C1' -B 500 -H 1
python d-dash_hc.py -N 'FNN' -R 300 -T True -F 'FNN_with_target_net_C1' -B 500 -H 1
python d-dash_hc.py -N 'LSTM' -R 300 -T True -F 'LSTM_with_target_net_C1' -B 500 -H 1

# change the channel history to 4
python d-dash_hc.py -N 'FNN' -R 300 -T False -F 'FNN_without_target_net_C4' -B 500 -H 4
python d-dash_hc.py -N 'LSTM' -R 300 -T False -F 'LSTM_without_target_net_C4' -B 500 -H 4
python d-dash_hc.py -N 'FNN' -R 300 -T True -F 'FNN_with_target_net_C4' -B 500 -H 4
python d-dash_hc.py -N 'LSTM' -R 300 -T True -F 'LSTM_with_target_net_C4' -B 500 -H 4

# change hidden layer number
# change the hidden layer to the half
python d-dash_hc.py -N 'FNN' -R 300 -T False -F 'FNN_without_target_net_hid' -B 500 -L 64
python d-dash_hc.py -N 'LSTM' -R 300 -T False -F 'LSTM_without_target_net_hid' -B 500 -L 64
python d-dash_hc.py -N 'FNN' -R 300 -T True -F 'FNN_with_target_net_hid' -B 500 -L 64
python d-dash_hc.py -N 'LSTM' -R 300 -T True -F 'LSTM_with_target_net_hid' -B 500 -L 64

# change episode
# change the episode to 100
python d-dash_hc.py -N 'FNN' -R 100 -T False -F 'FNN_without_target_net_E100' -B 500
python d-dash_hc.py -N 'LSTM' -R 100 -T False -F 'LSTM_without_target_net_E100' -B 500
python d-dash_hc.py -N 'FNN' -R 100 -T True -F 'FNN_with_target_net_E100' -B 500
python d-dash_hc.py -N 'LSTM' -R 100 -T True -F 'LSTM_with_target_net_E100' -B 500

# change the episode to 500
python d-dash_hc.py -N 'FNN' -R 500 -T False -F 'FNN_without_target_net_E500' -B 500
python d-dash_hc.py -N 'LSTM' -R 500 -T False -F 'LSTM_without_target_net_E500' -B 500
python d-dash_hc.py -N 'FNN' -R 500 -T True -F 'FNN_with_target_net_E500' -B 500
python d-dash_hc.py -N 'LSTM' -R 500 -T True -F 'LSTM_with_target_net_E500' -B 500

# plot and save it to "./fig/"
python plot.py

clear
pyfiglet -f slant "Finished"
echo "images have been saved in ./fig"
