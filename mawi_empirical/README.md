```markdown
# MAWI Empirical Validation High-Fidelity Backbone Census

## Module Introduction and Data Source
This module processes real-world high-throughput backbone network traffic data. The core objective is to validate the latency-dominated physical main sequence and the rate paradox gap within the RoS-ETA framework. Our experimental data is sourced from the MAWI archive of the WIDE Project. The specific full snapshot dataset covering 15 minutes on February 1 2026 is available at https://mawi.wide.ad.jp/mawi/samplepoint-F/2026/202602011400.html



Due to the massive volume of real backbone traffic the aforementioned 15-minute peak snapshot contains nearly **177 million** packets. Traditional one-time memory loading methods inevitably cause severe memory overflow crashes. We designed an industrial-grade large-flow data processing pipeline adopting a divide-and-conquer strategy that combines streaming fragmentation batch extraction and incremental aggregation plotting. This enables the feature extraction and high-fidelity physical validation of hundreds of millions of packets on a standard personal computer.

## Environment and Dependency Preparation
Running this module requires underlying network packet analysis tools and a Python runtime environment.

First install Wireshark and ensure its command-line tools editcap.exe and tshark.exe are configured in the system environment variables or provide their absolute paths when running subsequent scripts.

A Python environment of version 3.8 or higher is recommended. Before executing the code fully install the following core dependencies via pip
* **pandas** handles chunked reading and aggregation of massive CSV data
* **numpy** executes numerical computations in logarithmic space
* **matplotlib** renders high-fidelity physical manifold plots containing tens of thousands of scatter points
* **scikit-learn** invokes RANSACRegressor for robust linear fitting

Generating massive temporary files and massive CSV result files is an inherent part of the process. Ensure the executing hard drive has at least **50GB** of available space and **16GB** of physical system RAM.

## Step 1 Industrial-Grade Streaming Split
We slice the multi-gigabyte raw pcap archives into small easily swallowable files. A packet-count-based splitting approach ensures stability when processing massive files. Create a temporary folder mawi_chunks in the data directory to store the sliced files. Use editcap to slice the large file into small files containing two million packets each.

```powershell
mkdir mawi_chunks
cd E:\project_learn\RoS-ETA\mawi
& "C:\Program Files\Wireshark\editcap.exe" -c 2000000 "202602011400.pcap.gz" mawi_chunks\mawi_part.pcap

```

This splitting process may take a considerable amount of time. It will generate a large number of sequentially numbered pcap slice files in the newly created directory.

## Step 2 Batch Sanitization and Feature Extraction

We wrote an automated PowerShell script process_all.ps1 to iterate through all files circumventing the manual processing of hundreds of slice files. This script automatically performs VLAN header sanitization extracts crucial TCP five-tuple and spatiotemporal features and appends all results to a global CSV file.

```powershell
$tshark_path = "C:\Program Files\Wireshark\tshark.exe"
$editcap_path = "C:\Program Files\Wireshark\editcap.exe"
$source_dir = ".\mawi_chunks"
$output_csv = "mawi_full_day.csv"

"src_ip,dst_ip,src_port,dst_port,timestamp,len" | Out-File -Encoding UTF8 $output_csv
$files = Get-ChildItem "$source_dir\mawi_part_*.pcap"

foreach ($file in $files) {
    Write-Host "Processing $($file.Name)"
    $temp_clean = "$source_dir\temp_clean.pcap"
    & $editcap_path -D 64 $file.FullName $temp_clean
    & $tshark_path -r $temp_clean -Y "tcp.len > 0" -T fields -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e frame.time_epoch -e frame.len -E header=n -E "separator=," | Out-File -Encoding UTF8 -Append $output_csv
    Remove-Item $temp_clean
}
Write-Host "Extraction Complete"

```

Running this script in unrestricted mode in PowerShell automatically completes all extraction workloads and cleans up temporary files to free up disk space.

## Step 3 Big Data Incremental Aggregation and Physical Plotting

Handling the generated massive CSV file is the primary task of this phase. The chunksize blocked reading technique of Pandas effortlessly ingests gigabytes of data. The Python script aggregates bidirectional flows and calculates physical features including duration total bytes and micro-entropy proxies before ultimately fitting and plotting the physical manifold graph.

```bash
python verify_mawi_physics.py

```

The complete plotting logic and data filtering rules are directly available in the validation source code within the project repository.

## Scientific Findings and Result Evaluation

We conducted high-fidelity physical validations on this 15-minute real backbone snapshot through the aforementioned data pipeline. Based on the model extraction and statistical results we can summarize the following core profile of this dataset

* Traffic capture duration **899.96** seconds
* Total packet count **177,664,523** packets
* Average network bandwidth during the period **1.42** Gbps
* Valid long flows with complete interactions extracted **51,568** flows

The experimental results compellingly demonstrate macroscopic statistical stability under massive traffic. Even at extreme scales covering 10GB-level elephant flows these tens of thousands of real benign flows tightly converge on the main sequence band with a slope of 0.33. This empirically confirms that the vast majority of backbone traffic is constrained by TCP handshakes and RTT latency completely incapable of touching the physical link line rate.

Simultaneously the data intuitively reveals the absolute rigidity of physical boundaries. A massive rate paradox gap exists between the algorithmically fitted benign manifold and the theoretical physical limit. Among more than fifty thousand real traffic samples less than **0.05%** of isolated points touch the limit line. This endows RoS-ETA with extremely high physical defense confidence proving that any network attack attempting to hug the physical limit to maintain destructive power will be exposed as an explicit geometric outlier within this feature space.

```

```
