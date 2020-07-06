# vector_multi_matrix

## 程式說明
這是向量乘矩陣的運算，使用1維方式傳入傳出資料

## 檔案內容
* main.c 主程式
* kernel_program.cl OpenCL核心程式
* Makefile 編譯腳本
* run.bash 執行腳本

## 編譯方法
```bash
make #編譯release版本
make ver=debug #編譯debug版本
make clean #清除編譯結果
```
> 編譯結果在 ../bin/ 路徑下

## 執行程式
```bash
sudo bash run.bash 
```