gpu_freq=$1
cur_freq=$(cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq)

#Change GPU Frequency
if [ $gpu_freq -gt $cur_freq ];
then
	sudo bash -c 'echo '${gpu_freq}' >  /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq'
	sudo bash -c 'echo '${gpu_freq}' >  /sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq'
else 
	sudo bash -c 'echo '${gpu_freq}' >  /sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq'
	sudo bash -c 'echo '${gpu_freq}' >  /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq'
fi
