
# **Robust Weight Transfer for Maya**

Weight transfer code is based on https://github.com/rin-23/RobustSkinWeightsTransferCode/tree/main

## Dependencies
- numpy
- scipy

```bat
mayapy -m pip install numpy
mayapy -m pip install scipy
```

## Usage
![image](https://raw.githubusercontent.com/wlee445/maya-robust-weight-transfer/main/data/image/RobustWeightTransferUI.png)
### startup
#### Use `Maya2025.bat` start maya
```
import weight_transfer.ui as weight_transfer_ui
weight_transfer_ui.show()
```
#### Or add `robust-weight-transfer.mod` path copy to `MAYA_MODULE_PATH`
Maya module help on [Distributing plug-ins using modules](https://help.autodesk.com/view/MAYADEV/2025/CHS/?guid=Maya_DEVHELP_Distributing_Maya_Plug_ins_DistributingUsingModules_ModuleDescriptionFiles_html)



## Citation
If you use this unofficial implementation in your work, please cite the original paper as follows:
```bibtex
@inproceedings{abdrashitov2023robust,
author = {Abdrashitov, Rinat and Raichstat, Kim and Monsen, Jared and Hill, David},
title = {Robust Skin Weights Transfer via Weight Inpainting},
year = {2023},
isbn = {9798400703140},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3610543.3626180},
doi = {10.1145/3610543.3626180},
booktitle = {SIGGRAPH Asia 2023 Technical Communications},
articleno = {25},
numpages = {4},
location = {<conf-loc>, <city>Sydney</city>, <state>NSW</state>, <country>Australia</country>, </conf-loc>},
series = {SA '23}
}

```