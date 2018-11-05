# possible configuration space
BATCH_SIZE_ARRAY=(1 8 16 32)
ALL_MEM_GROWTH_ARRAY=([1]=0.1 [1]=0.15 [2]=0.25 [3]=0.33)
GPU_FREQ_ARRAY=([1]=76800000 [2]=537600000 [3]=998400000)
CPU_FREQ_ARRAY=([1]=102000 [2]=918000 [3]=1734000)
EMC_FREQ_ARRAY=([1]=12750000 [2]=800000000 [3]=1600000000)
DISABLE_CORE_ARRAY=([1]=0 [2]=1 [3]=2 [4]=3)

sh ./cpu_freq.sh ${DISABLE_CORE_ARRAY[4]} ${CPU_FREQ_ARRAY[1]}
sh ./gpu_freq.sh ${GPU_FREQ_ARRAY[1]}
sh ./emc_freq.sh ${EMC_FREQ_ARRAY[1]}

#python run_benchmark.py --batch_size 32 --all_mem_gr 0 mem_fr_per_gpu 0.25

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