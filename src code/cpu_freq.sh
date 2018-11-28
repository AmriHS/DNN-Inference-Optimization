num_cores=$1
cpu_freq=$2
cur_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)

# Disable/enable CPU cores
# 3 is the total number of cores we are able to enable.
# We start from core 1 to num_cores requested to disable. Subsequently, we enable cores that are not asked for. 

for i in $(seq 1 1 $num_cores)
	do
		sudo bash -c 'echo 0 > /sys/devices/system/cpu/cpu['$i']/online'
	done

num_cores=$((num_cores+1))
for i in $(seq $num_cores 1 3) 
	do
		sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpu['$i']/online'
	done


#Change GPU Frequency
if [ ! -z $cpu_freq ];
then
	if [ $cpu_freq -gt $cur_freq ];
	then
		sudo bash -c 'echo '${cpu_freq}' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq'
		sudo bash -c 'echo '${cpu_freq}' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq'
		
	else
		sudo bash -c 'echo '${cpu_freq}' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq'
		sudo bash -c 'echo '${cpu_freq}' > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq'
	fi
fi
