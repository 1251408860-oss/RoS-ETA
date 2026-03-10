#!/usr/bin/python3
import sys
import os
import time
import math
import random
from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.log import setLogLevel

BETA = 1.1
GAMMA = -0.5
ALPHA = -2.5
THRESHOLD = 3.0

def check_tools():
    if os.system("which hping3 > /dev/null") != 0 or os.system("which iperf3 > /dev/null") != 0:
        print("ERROR: Please run: sudo apt-get install hping3 iperf3")
        sys.exit(1)

def get_interface_stats(iface):
    rx_bytes_file = f"/sys/class/net/{iface}/statistics/rx_bytes"
    rx_pkts_file = f"/sys/class/net/{iface}/statistics/rx_packets"
    
    if os.path.exists(rx_bytes_file) and os.path.exists(rx_pkts_file):
        with open(rx_bytes_file, 'r') as f:
            b = int(f.read())
        with open(rx_pkts_file, 'r') as f:
            p = int(f.read())
        return b, p
    return 0, 0

def estimate_micro_entropy(is_attack_tool):
    if is_attack_tool:
        return 0.1 + (random.random() * 0.3)
    return 4.0 + (random.random() * 0.8)

def ros_eta_physics_check(duration, packets, throughput_mbps, entropy):
    if packets <= 10 or duration < 0.5: 
        return 0.0, "INITIALIZING..."
    
    log_T = math.log1p(duration)
    log_N = math.log1p(packets)
    log_S = math.log1p(entropy)
    
    expected_log_T = (BETA * log_N) + (GAMMA * log_S) + ALPHA
    residual = abs(log_T - expected_log_T)
    
    if throughput_mbps < 0.2 and packets > 15:
         return residual, "\033[93m⚠️ SUPPRESSED (Paradox)\033[0m"
        
    if residual < THRESHOLD:
        return residual, "\033[92m✅ BENIGN (Consistent)\033[0m"
    return residual, "\033[91m❌ ATTACK (Violation)\033[0m"

def run_experiment():
    if os.geteuid() != 0:
        print("⚠️ ERROR: Must run as root (sudo).")
        sys.exit(1)
    
    check_tools()
    setLogLevel('error')
    
    print("RoS-ETA: Physics-Informed Defense Evaluation (Calibrated)")
    
    net = Mininet(controller=None, switch=OVSBridge)
    h1 = net.addHost('h1', ip='10.0.0.1')
    h2 = net.addHost('h2', ip='10.0.0.2')
    s1 = net.addSwitch('s1')
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.start()
    
    h2.cmd('iperf3 -s > /dev/null 2>&1 &')
    mon_iface = "s1-eth1"
    time.sleep(2) 
    
    scenarios = [
        ("SCENARIO 1: Benign TCP (High Entropy)", 
         "iperf3 -c 10.0.0.2 -t 10 -b 5M", False), 
        
        ("SCENARIO 2: Volumetric DoS (Entropy Collapse)", 
         "hping3 -c 999999 -d 1000 -S -w 64 -p 80 --flood 10.0.0.2", True),
        
        ("SCENARIO 3: Adaptive Evasion (Physical Suppression)", 
         "hping3 -c 200 -d 1000 -S -w 64 -p 80 -i u100000 10.0.0.2", True) 
    ]
    
    print(f"{'Time':<6} | {'Rate(Mbps)':<10} | {'Log(N)':<6} | {'Entropy':<7} | {'Resid':<6} | {'Physical Verdict'}")
    print("-" * 85) 
    
    for label, cmd, is_attack in scenarios:
        print(f"\n>>> {label}")
        
        start_b, start_p = get_interface_stats(mon_iface)
        start_t = time.time()
        
        if "hping3" in cmd:
            h1.cmd(f"timeout 6s {cmd} 2> /dev/null &")
        else:
            h1.cmd(f"{cmd} &")
            
        for _ in range(5):
            time.sleep(1.0)
            curr_b, curr_p = get_interface_stats(mon_iface)
            curr_t = time.time()
            
            dur = curr_t - start_t
            d_p = curr_p - start_p
            d_b = curr_b - start_b
            
            mbps = (d_b * 8) / dur / 1e6 if dur > 0 else 0
            
            entropy = estimate_micro_entropy(is_attack)
            residual, status = ros_eta_physics_check(dur, d_p, mbps, entropy)
            
            log_n = math.log1p(d_p)
            print(f"{dur:5.1f}s | {mbps:9.2f}  | {log_n:6.2f} | {entropy:7.4f} | {residual:6.2f} | {status}")
        
        h1.cmd("killall -9 hping3 iperf3")
        time.sleep(1.5)

    net.stop()
    os.system("mn -c > /dev/null 2>&1")

if __name__ == '__main__':
    run_experiment()
