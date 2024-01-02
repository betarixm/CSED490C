# Assignment 4. 2D Convolution

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
- BogoMIPS: 7195.29
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

- Static hostname: n4
- Icon name: computer-desktop
- Chassis: desktop
- Machine ID: f05f160400af47df81f9fdcaa30ff384
- Boot ID: ee60c106f2ff4403a3f329c12084093b
- Operating System: CentOS Linux 7 (Core)
- CPE OS Name: cpe
- Kernel: Linux 3.10.0-1160.83.1.el7.x86_64
- Architecture: x86-64

## Results by Data

|Data|Correctness|Doing GPU memory allocation|Copying data to the GPU|Doing the computation on the GPU|Copying data from the GPU|Doing GPU Computation (memory + compute)|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|True|6.22228 ms|0.094458 ms|0.092017 ms|0.107679 ms|6.8115 ms|
|1|True|0.187977 ms|0.072196 ms|0.070399 ms|0.124763 ms|0.627459 ms|
|2|True|0.15346 ms|0.243707 ms|0.157595 ms|0.560615 ms|1.26319 ms|
|3|True|0.505209 ms|0.533552 ms|0.303382 ms|1.89438 ms|3.51632 ms|
|4|True|0.691853 ms|2.22639 ms|1.85677 ms|5.10953 ms|10.1166 ms|
|5|True|0.743653 ms|12.5424 ms|14.1466 ms|25.5858 ms|53.1949 ms|
|6|True|1.61651 ms|81.1907 ms|112.721 ms|165.62 ms|361.298 ms|
