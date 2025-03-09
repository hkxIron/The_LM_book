#!/usr/bin/env bash

show_run_time(){
  local start_time=$1
  local end_time=$2
  local elapsed_time=$((end_time - start_time))
  # 转换为分钟和秒数
  local minutes=$((elapsed_time / 60))
  local seconds=$((elapsed_time % 60))
  echo "All job done!脚本已经运行了 ${minutes} 分钟 ${seconds} 秒。"
  echo `date`
}

test_max_gpu_num(){
  local max_gpu_num=$1 # 最少多少个gpu
  echo "限制最大gpu数量：$max_gpu_num"
  seq 0 7 >tmp1.txt
  nvidia-smi|grep "Processes:" -A14|grep "MiB"|awk -F' ' '{if(length($2)>0) {print $2} }'>tmp2.txt
  device_list=$(sort -n tmp1.txt tmp2.txt tmp2.txt|uniq -u|head -n $max_gpu_num|tr '\n' ','|sed 's/,$//');rm -f tmp1.txt tmp2.txt
  gpu_num=$(echo ${device_list}|awk -F',' '{print NF}')
  echo "gpu ids:$device_list size:${gpu_num}"
  if [ -z "${device_list}" ];then
    echo "========================警告：没有可用gpu,退出======================="
    exit 100
  fi
}

