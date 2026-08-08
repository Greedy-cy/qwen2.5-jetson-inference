#!/bin/bash
# 锁定 Jetson Orin 到基准测试状态:
#   GPU  -> MAXN_SUPER 模式 + jetson_clocks (最高频率)
#   EMC  -> 手动锁 bpmp 到硬件上限 3199MHz (jetson_clocks 只会锁到
#           nvpmodel cap 2133MHz, 必须额外用 bpmp 锁高频)
# 用法: ./scripts/lock_clocks.sh   (需要 sudo 密码)
set -e

echo "== nvpmodel MAXN_SUPER =="
sudo nvpmodel -m 2

echo "== jetson_clocks (CPU/GPU/EMC -> cap) =="
sudo jetson_clocks

echo "== EMC -> 3199MHz (bpmp 硬锁) =="
sudo bash -c 'echo 3199000000 > /sys/kernel/debug/bpmp/debug/clk/emc/rate && echo 1 > /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked'

echo ""
echo "GPU : $(cat /sys/class/devfreq/17000000.gpu/cur_freq) Hz (期望 1020000000)"
echo "EMC : $(sudo cat /sys/kernel/debug/bpmp/debug/clk/emc/rate) Hz (期望 3199000000)"
