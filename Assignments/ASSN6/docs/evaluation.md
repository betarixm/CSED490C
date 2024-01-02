# Assignment 6. List Scan

> Machine generated report

## Environment

### CPU

- Architecture: x86_64
- CPU op-mode(s): 32-bit, 64-bit
- Byte Order: Little Endian
- CPU(s): 6
- On-line CPU(s) list: 0-5
- Thread(s) per core: 1
- Core(s) per socket: 6
- Socket(s): 1
- NUMA node(s): 1
- Vendor ID: GenuineIntel
- CPU family: 6
- Model: 79
- Model name: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz
- Stepping: 1
- CPU MHz: 1200.000
- CPU max MHz: 3601.0000
- CPU min MHz: 1200.0000
- BogoMIPS: 7195.97
- Virtualization: VT-x
- L1d cache: 32K
- L1i cache: 32K
- L2 cache: 256K
- L3 cache: 15360K
- NUMA node0 CPU(s): 0-5
- Flags: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch epb cat_l3 cdp_l3 invpcid_single intel_ppin rsb_ctxsw tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm rdt_a rdseed adx smap intel_pt xsaveopt cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts

### GPU

- name: NVIDIA TITAN Xp
- memory.total [MiB]: 12288 MiB

### Others

- Static hostname: n6
- Icon name: computer-desktop
- Chassis: desktop
- Machine ID: f05f160400af47df81f9fdcaa30ff384
- Boot ID: e60e9a33da794c05b6f2f8412f10c1ba
- Operating System: CentOS Linux 7 (Core)
- CPE OS Name: cpe
- Kernel: Linux 3.10.0-1160.83.1.el7.x86_64
- Architecture: x86-64

## Results by Data

|Data|Correctness|Importing data and creating memory on host|Allocating GPU memory.|Clearing output memory.|Copying input memory to the GPU.|Performing CUDA computation|Copying output memory to the CPU|Freeing GPU Memory|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|True|5.21626 ms|0.194541 ms|0.032792 ms|0.043791 ms|0.064253 ms|0.029702 ms|0.145849 ms|
|1|True|4.15263 ms|0.16734 ms|0.025186 ms|0.038426 ms|0.053315 ms|0.023685 ms|0.112662 ms|
|2|True|11.6644 ms|0.195908 ms|0.0325 ms|0.035687 ms|0.063894 ms|0.030044 ms|0.140826 ms|
|3|True|58.8967 ms|0.158364 ms|0.027512 ms|0.039654 ms|0.056789 ms|0.034207 ms|0.117071 ms|
|4|True|30.3537 ms|0.180992 ms|0.029669 ms|0.035358 ms|0.060537 ms|0.030186 ms|0.129988 ms|
|5|True|92.0749 ms|0.146522 ms|0.045271 ms|0.04349 ms|0.039833 ms|0.036933 ms|0.12388 ms|
|6|True|197.015 ms|0.167869 ms|0.046772 ms|0.055056 ms|0.045437 ms|0.049707 ms|0.112177 ms|
|7|True|356.676 ms|0.136771 ms|0.038743 ms|0.10481 ms|0.069988 ms|0.099259 ms|0.094378 ms|
|8|True|495.272 ms|0.119726 ms|0.03832 ms|0.124795 ms|0.08012 ms|0.112372 ms|0.09697 ms|
|9|True|911.679 ms|0.11607 ms|0.035522 ms|0.24163 ms|0.139145 ms|0.207064 ms|0.087141 ms|
