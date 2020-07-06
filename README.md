# opencl_matrix_multi
OpenCL 幾個範例

## 開發環境
### clinfo 查看GPU設備
```bash
sudo apt install clinfo
```
### OpenCL library
```bash
sudo apt install ocl-icd-libopencl1
sudo apt install opencl-headers
```
### OpenCL debug 
```bash
sudo apt install ocl-icd-opencl-dev
```

## 範例說明
### matrix5_dot_matrix5
這是2個5*5方陣去做dot,使用2維方式傳入傳出資料

### vector_multi_matrix
這是向量乘矩陣的運算，使用1維方式傳入傳出資料

