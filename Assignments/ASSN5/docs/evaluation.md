# Assignment 5. Histogram

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
- BogoMIPS: 7196.50
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

- Static hostname: n1
- Icon name: computer-desktop
- Chassis: desktop
- Machine ID: f05f160400af47df81f9fdcaa30ff384
- Boot ID: b8782177d39b4cef98a02c7f2b1f495f
- Operating System: CentOS Linux 7 (Core)
- CPE OS Name: cpe
- Kernel: Linux 3.10.0-1160.83.1.el7.x86_64
- Architecture: x86-64

## Results by Data

|Data|Correctness|Importing data and creating memory on host|Allocating GPU memory.|Copying input memory to the GPU.|Performing CUDA computation|Copying output memory to the CPU|Freeing GPU Memory|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|True|7.77726 ms|4.04955 ms|0.066507 ms|0.088362 ms|0.06609 ms|0.20957 ms|
|1|True|11.0837 ms|0.194113 ms|0.048447 ms|0.058412 ms|0.043754 ms|0.141117 ms|
|2|True|10.3002 ms|0.314297 ms|0.067468 ms|0.102554 ms|0.061221 ms|0.197523 ms|
|3|True|9.26717 ms|0.154867 ms|0.036716 ms|0.047392 ms|0.028227 ms|0.116229 ms|
|4|True|4.76527 ms|0.201071 ms|0.05792 ms|0.058936 ms|0.046651 ms|0.143196 ms|
|5|True|1888.58 ms|0.127897 ms|0.357257 ms|0.055387 ms|0.020974 ms|0.088828 ms|
|6|True|2758.71 ms|0.248516 ms|0.489058 ms|0.068171 ms|0.027692 ms|0.222423 ms|
|7|True|4136.44 ms|0.259809 ms|0.634889 ms|0.085325 ms|0.023676 ms|0.279496 ms|
