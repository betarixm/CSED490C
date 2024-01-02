# Assignment 7. SpMV with JDS format

> Machine generated report

## Environment

### CPU

- Architecture: x86_64
- CPU op-mode(s): 32-bit, 64-bit
- Byte Order: Little Endian
- CPU(s): 12
- On-line CPU(s) list: 0-11
- Thread(s) per core: 2
- Core(s) per socket: 6
- Socket(s): 1
- NUMA node(s): 1
- Vendor ID: GenuineIntel
- CPU family: 6
- Model: 79
- Model name: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz
- Stepping: 1
- CPU MHz: 3704.589
- CPU max MHz: 4000.0000
- CPU min MHz: 1200.0000
- BogoMIPS: 7195.72
- Virtualization: VT-x
- L1d cache: 32K
- L1i cache: 32K
- L2 cache: 256K
- L3 cache: 15360K
- NUMA node0 CPU(s): 0-11
- Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb cat_l3 cdp_l3 invpcid_single intel_ppin rsb_ctxsw tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm rdt_a rdseed adx smap intel_pt xsaveopt cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts

### GPU

- name: Quadro P400
- memory.total [MiB]: 2048 MiB

### Others

- Static hostname: cseedu-master
- Icon name: computer-desktop
- Chassis: desktop
- Machine ID: 7c878f56bf2c4fc19460dc811d7e39a2
- Boot ID: 613f66d3b879420aa334fe56acd4cbaa
- Operating System: CentOS Linux 7 (Core)
- CPE OS Name: cpe
- Kernel: Linux 3.10.0-1160.83.1.el7.x86_64
- Architecture: x86-64

## Results by Data (Not Using Shared Memory)

|Data|Correctness|Importing data and creating memory on host|Allocating GPU memory.|Copying input memory to the GPU.|Performing CUDA computation|Copying output memory to the CPU|Freeing GPU Memory|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|True|0.312991 ms|0.128177 ms|0.044277 ms|0.044954 ms|0.01907 ms|0.099942 ms|
|1|True|1.3694 ms|0.135753 ms|0.054638 ms|0.06309 ms|0.018828 ms|0.105384 ms|
|2|True|2.40585 ms|0.137205 ms|0.063382 ms|0.088986 ms|0.01975 ms|0.10649 ms|
|3|True|1.75199 ms|0.135701 ms|0.056063 ms|0.058592 ms|0.018859 ms|0.106977 ms|
|4|True|6.56099 ms|0.140683 ms|0.084526 ms|0.105164 ms|0.017442 ms|0.108435 ms|
|5|True|52.0231 ms|0.146018 ms|0.409364 ms|0.330872 ms|0.017737 ms|0.112361 ms|
|6|True|200.885 ms|0.375914 ms|0.970626 ms|0.66839 ms|0.019751 ms|0.475746 ms|
|7|True|397.951 ms|0.389166 ms|1.81887 ms|1.28011 ms|0.01993 ms|0.81598 ms|
|8|True|6071.41 ms|0.538069 ms|24.5025 ms|15.2239 ms|0.023089 ms|9.43689 ms|

## Results by Data (Using Shared Memory)

|Data|Correctness|Importing data and creating memory on host|Allocating GPU memory.|Copying input memory to the GPU.|Performing CUDA computation|Copying output memory to the CPU|Freeing GPU Memory|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|True|0.320404 ms|0.126672 ms|0.043511 ms|0.044475 ms|0.018567 ms|0.101992 ms|
|1|True|1.37402 ms|0.128806 ms|0.052736 ms|0.063325 ms|0.018884 ms|0.101696 ms|
|2|True|2.36305 ms|0.126113 ms|0.060108 ms|0.089324 ms|0.019481 ms|0.101628 ms|
|3|True|1.72717 ms|0.136584 ms|0.057205 ms|0.058104 ms|0.019217 ms|0.105594 ms|
|4|True|6.56405 ms|0.133487 ms|0.082613 ms|0.105141 ms|0.018088 ms|0.10429 ms|
|5|True|51.9121 ms|0.141332 ms|0.406956 ms|0.335576 ms|0.017913 ms|0.112041 ms|
|6|True|200.732 ms|0.358056 ms|0.968464 ms|0.667724 ms|0.018998 ms|0.469002 ms|
|7|True|399.06 ms|0.374142 ms|1.82939 ms|1.29645 ms|0.019524 ms|0.815232 ms|
|8|True|6086.24 ms|0.566999 ms|24.4701 ms|15.2639 ms|0.022722 ms|9.57356 ms|
