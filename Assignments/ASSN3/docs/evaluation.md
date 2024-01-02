# Assignment 3. Vector Add with CUDA Streams and Pinned Memory

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
- BogoMIPS: 7195.88
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

- Static hostname: n8
- Icon name: computer-desktop
- Chassis: desktop
- Machine ID: f05f160400af47df81f9fdcaa30ff384
- Boot ID: fd4b03eefbf24d69af159452763c9eeb
- Operating System: CentOS Linux 7 (Core)
- CPE OS Name: cpe
- Kernel: Linux 3.10.0-1160.83.1.el7.x86_64
- Architecture: x86-64

## Results by Data

|Data|Correctness|Importing data and creating memory on host|Performing CUDA computation|Freeing Pinned Memory|
|:-:|:-:|:-:|:-:|:-:|
|0|True|1.63874 ms|0.466984 ms|0.538834 ms|
|1|True|3.31773 ms|0.255063 ms|0.293143 ms|
|2|True|4.75059 ms|0.205194 ms|0.254448 ms|
|3|True|6.32941 ms|0.197028 ms|0.21859 ms|
|4|True|9.42369 ms|0.191218 ms|0.271807 ms|
|5|True|75.9913 ms|0.196598 ms|0.340823 ms|
|6|True|105.228 ms|0.212755 ms|0.328713 ms|
|7|True|182.358 ms|0.196884 ms|0.311927 ms|
|8|True|339.359 ms|0.2203 ms|0.309982 ms|
|9|True|791.523 ms|0.263739 ms|0.34199 ms|

## Results by the Number of Streams

|Number of Streams|Correctness|Importing data and creating memory on host|Performing CUDA computation|Freeing Pinned Memory|
|:-:|:-:|:-:|:-:|:-:|
|1|True|746.618 ms|0.13745 ms|0.307976 ms|
|2|True|729.587 ms|0.140038 ms|0.307031 ms|
|3|True|735.354 ms|0.136876 ms|0.306211 ms|
|4|True|734.382 ms|0.149847 ms|0.319453 ms|
|5|True|739.29 ms|0.155863 ms|0.331489 ms|
|6|True|672.599 ms|0.162964 ms|0.305406 ms|
|7|True|672.264 ms|0.158396 ms|0.321882 ms|
|8|True|823.269 ms|0.173934 ms|0.321687 ms|
|9|True|693.038 ms|0.18305 ms|0.317499 ms|
|10|True|737.105 ms|0.181003 ms|0.335607 ms|
|11|True|673.904 ms|0.190667 ms|0.318196 ms|
|12|True|672.323 ms|0.178361 ms|0.326133 ms|
|13|True|752.41 ms|0.197107 ms|0.332082 ms|
|14|True|809.725 ms|0.195769 ms|0.336507 ms|
|15|True|675.872 ms|0.20466 ms|0.324709 ms|
|16|True|808.632 ms|0.256545 ms|0.34218 ms|
|17|True|692.821 ms|0.255252 ms|0.328139 ms|
|18|True|678.131 ms|0.260138 ms|0.323823 ms|
|19|True|818.187 ms|0.269758 ms|0.349887 ms|
|20|True|776.22 ms|0.291203 ms|0.366936 ms|
|21|True|673.209 ms|0.27979 ms|0.328515 ms|
|22|True|672.556 ms|0.298701 ms|0.337517 ms|
|23|True|718.995 ms|0.28767 ms|0.343073 ms|
|24|True|704.47 ms|0.292245 ms|0.337995 ms|
|25|True|690.441 ms|0.301499 ms|0.348236 ms|
|26|True|815.74 ms|0.325358 ms|0.34917 ms|
|27|True|679.052 ms|0.316948 ms|0.342557 ms|
|28|True|732.622 ms|0.331249 ms|0.361371 ms|
|29|True|747.725 ms|0.339496 ms|0.367599 ms|
|30|True|677.825 ms|0.327674 ms|0.347158 ms|
|31|True|673.815 ms|0.353729 ms|0.343227 ms|
|32|True|747.107 ms|0.364926 ms|0.361939 ms|
