# MAWI Empirical Validation High-Fidelity Backbone Census

## Module Overview and Data Provenance

This module is primarily designed to process real-world high-throughput backbone traffic data. The core objective is to validate the latency-dominated physical main sequence and the rate paradox gap within the RoS-ETA framework. Our experimental data is sourced from the MAWI archive of the WIDE project. The specific 15-minute full snapshot dataset from February 1 2026 is available at [https://mawi.wide.ad.jp/mawi/samplepoint-F/2026/202602011400.html](https://mawi.wide.ad.jp/mawi/samplepoint-F/2026/202602011400.html)

Given the sheer volume of genuine backbone data where the aforementioned 15-minute peak snapshot alone contains nearly **177 million** packets traditional one-time memory loading methods inevitably trigger severe Out-Of-Memory crashes. To overcome this we engineered an industrial-grade large flow data processing pipeline. By adopting a divide-and-conquer strategy that combines streaming slicing batch extraction and incremental aggregation plotting this solution successfully accomplishes feature extraction and high-fidelity physical validation for hundreds of millions of packets within a standard personal computing environment.

## Environment and Dependency Preparation

Running this module requires the prior setup of underlying network packet analysis tools and a Python runtime environment.

You must first install the Wireshark software and ensure its command-line utilities editcap.exe and tshark.exe are configured in the system environment variables or provide their absolute paths when executing subsequent scripts.

A Python environment of version 3.8 or higher is recommended. Before running the code please fully install the following core dependencies via pip

* **pandas** deployed for chunked reading and aggregation of massive CSV data
* **numpy** utilized for numerical computations in the logarithmic space
* **matplotlib** used to render high-fidelity physical manifold plots containing tens of thousands of scatter points
* **scikit-learn** required to invoke RANSACRegressor for noise-resistant linear fitting

Furthermore since the processing workflow generates numerous temporary files and an exceptionally large CSV result file please ensure the executing hard drive has at least **50GB** of available space alongside **16GB** of system physical memory.

## Step 1 Industrial-Grade Streaming Slicing

We partition the multi-gigabyte raw pcap archive into manageable smaller files. A packet-count-based slicing approach is applied here to guarantee stability when handling massive files. First create a temporary directory named mawi_chunks under the data folder to store the sliced files. Next utilize editcap to divide the large file into smaller segments containing two million packets each.

```powershell
mkdir mawi_chunks
cd E:\project_learn\RoS-ETA\mawi
& "C:\Program Files\Wireshark\editcap.exe" -c 2000000 "202602011400.pcap.gz" mawi_chunks\mawi_part.pcap

```

This slicing procedure may take a considerable amount of time. Upon completion it generates a large number of sequentially numbered pcap slice files in the newly created directory.

## Step 2 Batch Sanitization and Feature Extraction

To bypass the manual handling of hundreds or thousands of sliced files we authored an automated PowerShell script process_all.ps1 to iterate through all files. This script automatically performs VLAN header sanitization and extracts crucial TCP five-tuple alongside spatial-temporal features ultimately appending all results into a global CSV file.

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

Running this script in PowerShell under the unrestricted execution policy will automatically complete all extraction workloads and dynamically clean up temporary files to free up disk space.

## Step 3 Big Data Incremental Aggregation and Physical Plotting

During this phase we need to process the resulting massive CSV file. Leveraging the chunksize block-reading technology of Pandas allows for effortless throughput of multi-gigabyte data. The Python script aggregates bidirectional flows and calculates physical features including flow duration total bytes and micro-entropy proxies to ultimately fit and render the physical manifold plot.

```bash
python verify_mawi_physics.py

```

The complete plotting logic and data filtering rules can be referenced directly in the validation source code within the project repository.

## Scientific Discoveries and Result Evaluation

Through the aforementioned data pipeline we conducted a high-fidelity physical validation on this 15-minute real-world backbone snapshot. Based on the model extraction and statistical results we can summarize the core profile of this dataset as follows

* Traffic capture duration **899.96** seconds
* Total number of encompassed packets **177,664,523**
* Average network bandwidth reached **1.42** Gbps
* Successfully extracted valid long flows containing complete interactions totaling **51,568**

The experimental results compellingly demonstrate macro-statistical stability under massive traffic scales. Even at an extreme scale encompassing 10GB-level elephant flows these tens of thousands of genuine benign flows remain tightly converged on the main sequence band with a slope of 0.33. This empirically confirms that the vast majority of backbone traffic is constrained by TCP protocol handshakes and RTT latency completely incapable of reaching the physical link-layer line rate.

Simultaneously the data intuitively reveals the absolute rigidity of physical boundaries. A massive rate paradox gap exists between the algorithmically fitted benign manifold and the theoretical physical limit. Among over fifty thousand authentic traffic samples less than **0.05%** of isolated points touch the limit line. This endows RoS-ETA with exceptionally high physical defense confidence proving that any cyberattack attempting to approach physical limits to sustain destructive power will inevitably be exposed as explicit geometric outliers within this feature space.
