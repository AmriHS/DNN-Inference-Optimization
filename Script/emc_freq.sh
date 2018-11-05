emc_freq=$1
cur_freq=$(cat /sys/kernel/debug/clk/override.emc/clk_rate)

#Change EMC Frequency
sudo bash -c 'echo '${emc_freq}' > /sys/kernel/debug/clk/override.emc/clk_update_rate'
sudo bash -c 'echo 1 > /sys/kernel/debug/clk/override.emc/clk_state'
