# Assignment 1. Vector Addition

> Machine generated report

## Environment

### CPU

- Architecture: x86_64
- CPU op-mode(s): 32-bit, 64-bit
- Byte Order: Little Endian
- CPU(s): 20
- On-line CPU(s) list: 0-19
- Thread(s) per core: 1
- Core(s) per socket: 10
- Socket(s): 2
- NUMA node(s): 2
- Vendor ID: GenuineIntel
- CPU family: 6
- Model: 85
- Model name: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz
- Stepping: 7
- CPU MHz: 3077.840
- CPU max MHz: 3200.0000
- CPU min MHz: 1000.0000
- BogoMIPS: 4400.00
- Virtualization: VT-x
- L1d cache: 32K
- L1i cache: 32K
- L2 cache: 1024K
- L3 cache: 14080K
- NUMA node0 CPU(s): 0-9
- NUMA node1 CPU(s): 10-19
- Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req pku ospke avx512_vnni md_clear flush_l1d arch_capabilities

### GPU

- name: NVIDIA GeForce RTX 2080 Ti
- memory.total [MiB]: 11264 MiB

### Others

- Static hostname: n4.gasi-cluster
- Icon name: computer-server
- Chassis: server
- Machine ID: 29c6619de75542e99fb2934d6cccc8f9
- Boot ID: 298c3a6c1f5849bbbeddb79c619927ed
- Kernel: Linux 4.18.0-425.3.1.el8.x86_64
- Architecture: x86-64

## Elapsed Times

|Key|Correctness|Importing data and creating memory on host|Allocating GPU memory.|Copying input memory to the GPU.|Performing CUDA computation|Copying output memory to the CPU|Freeing GPU Memory|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|True|0.462124 ms|0.186625 ms|0.040698 ms|0.042 ms|0.018313 ms|0.167676 ms|
|1|True|0.896169 ms|0.181373 ms|0.044717 ms|0.041233 ms|0.017783 ms|0.161004 ms|
|2|True|1.23911 ms|0.185277 ms|0.041854 ms|0.040655 ms|0.019124 ms|0.164581 ms|
|3|True|1.50715 ms|0.188631 ms|0.04297 ms|0.042325 ms|0.019265 ms|0.165705 ms|
|4|True|12.8311 ms|0.185901 ms|0.043789 ms|0.040661 ms|0.019382 ms|0.161731 ms|
|5|True|101.878 ms|0.190003 ms|0.073299 ms|0.040056 ms|0.028338 ms|0.163939 ms|
|6|True|140.864 ms|0.188417 ms|0.083001 ms|0.041339 ms|0.031227 ms|0.165562 ms|
|7|True|235.945 ms|0.19274 ms|0.10642 ms|0.040881 ms|0.042229 ms|0.168594 ms|
|8|True|391.54 ms|0.184496 ms|0.160211 ms|0.040806 ms|0.06239 ms|0.1704 ms|
|9|True|714.309 ms|0.189738 ms|4.26865 ms|0.091104 ms|0.135777 ms|0.196487 ms|
