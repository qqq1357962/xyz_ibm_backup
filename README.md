# Global Placer

## Build
Install libtorch 1.11.0 + cpu
```
bash install_thirdparty.sh
```

```
bash build.sh
```
## Run
bash run_openroad.sh

## Parameter
--macro_padding: macro padding by site height
--utilization: std cell area utilization
--skip_patoh: skip the PaToh partition(if false 3d GP won't change the layer)
--stop_overflow_3d: Stop Threshold in 3d placment
--stop_overflow_via: Stop Threshold in HBT Placement
--bondingSizeX: bonding terminal size of x
--bondingSizeY: bonding terminal size of y
--bondingSpace: bonding terminal space between each other
