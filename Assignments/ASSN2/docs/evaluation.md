# Assignment 2. Tiled Matrix Multiplication

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
- BogoMIPS: 7196.33
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

- Static hostname: n7
- Icon name: computer-desktop
- Chassis: desktop
- Machine ID: f05f160400af47df81f9fdcaa30ff384
- Boot ID: 612969c56bb543b38673c66c7bce66f3
- Operating System: CentOS Linux 7 (Core)
- CPE OS Name: cpe
- Kernel: Linux 3.10.0-1160.83.1.el7.x86_64
- Architecture: x86-64

## Results by Data

|Data|Correctness|Importing data and creating memory on host|Allocating GPU memory.|Copying input memory to the GPU.|Performing CUDA computation|Copying output memory to the CPU|Freeing GPU Memory|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|0|True|3.85408 ms|0.394578 ms|0.755362 ms|0.095235 ms|0.044289 ms|0.203648 ms|
|1|True|8.24583 ms|0.329 ms|0.08645 ms|0.089482 ms|0.048892 ms|0.214037 ms|
|2|True|8.14669 ms|0.184635 ms|0.067755 ms|0.066742 ms|0.032913 ms|0.134182 ms|
|3|True|7.25252 ms|0.319974 ms|0.083678 ms|0.087814 ms|0.042818 ms|0.19743 ms|
|4|True|32.0709 ms|0.270698 ms|0.117348 ms|0.077983 ms|0.067686 ms|0.1667 ms|
|5|True|200.936 ms|0.293454 ms|0.538316 ms|0.211929 ms|0.772922 ms|0.21771 ms|
|6|True|434.242 ms|0.359498 ms|1.15881 ms|1.27742 ms|0.59197 ms|0.389372 ms|
|7|True|1016.01 ms|0.337858 ms|2.29985 ms|4.22994 ms|1.80899 ms|0.723379 ms|
|8|True|6788.94 ms|0.466941 ms|14.1388 ms|19.1326 ms|2.62632 ms|3.96938 ms|

## Results by Tile Width

|Tile Width|Correctness|Importing data and creating memory on host|Allocating GPU memory.|Copying input memory to the GPU.|Performing CUDA computation|Copying output memory to the CPU|Freeing GPU Memory|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|2|True|6720.2 ms|0.45744 ms|14.0006 ms|966.791 ms|2.25125 ms|4.02318 ms|
|4|True|6663.42 ms|0.481575 ms|13.8672 ms|126.754 ms|2.07454 ms|4.0061 ms|
|8|True|6851.85 ms|0.478291 ms|14.5978 ms|51.1917 ms|2.13252 ms|4.02708 ms|
|12|True|6579.27 ms|0.443593 ms|14.3293 ms|27.2375 ms|2.03083 ms|7.2442 ms|
|16|True|6425.46 ms|0.454709 ms|13.783 ms|19.2568 ms|2.20211 ms|8.99286 ms|
|24|True|6665.68 ms|0.452484 ms|13.905 ms|20.7359 ms|2.02374 ms|4.00482 ms|
|32|True|6665.01 ms|0.456231 ms|13.8738 ms|18.9403 ms|2.0848 ms|4.00535 ms|
