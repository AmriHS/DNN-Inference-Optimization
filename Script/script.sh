# possible configuration space
$CPU_FREQ = $1
$CPU_DIS_CORES = $2
$GPU_FREQ = $3
$EMC_FREQ = $4

sh ./cpu_freq.sh $CPU_FREQ $CPU_DIS_CORES
sh ./gpu_freq.sh $GPU_FREQ
sh ./emc_freq.sh $EMC_FREQ

python run_benchmark.py --bsize 32 --all_growth 1 --mem_frac 0.25

# verify configuration
cur_gpu_freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
cur_cpu_freq=$(cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq)
cur_emc_freq=$(cat /sys/kernel/debug/tegra_bwmgr/emc_rate)
dis_cpu_core_1=$(cat /sys/devices/system/cpu/cpu1/online)
dis_cpu_core_2=$(cat /sys/devices/system/cpu/cpu2/online)
dis_cpu_core_3=$(cat /sys/devices/system/cpu/cpu3/online)


echo "GPU Frequency: ${cur_gpu_freq}"
echo "CPU Frequency: ${cur_cpu_freq}"
echo "EMC Frequency: ${cur_emc_freq}"
echo "CPU 1 core Status: ${dis_cpu_core_1}"
echo "CPU 2 core Status: ${dis_cpu_core_2}"
echo "CPU 3 core Status: ${dis_cpu_core_3}"
#echo "$host, `date`, checkout,$Time_checkout" >> log.csv