# PMC

## Requirements:
* cmake (3.9 or higher),
* gcc, g++ (5.4.0 or higher),
*
## Build

Go to directory [PMC/PMC_source/](https://github.com/lstruski/PMC/tree/master/PMC_source) and run the following instruction in terminal:

```
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -G "CodeBlocks - Unix Makefiles" ..

cd ..

cmake --build ./build/ --target PMC -- -j 2
```
You run
```
cmake --build ./build/ --target clean -- -j 2
```
if you want to clean it.

## Demo

In folder [PMC/demo/](https://github.com/lstruski/PMC/tree/master/demo) I putted sample data together [results](https://github.com/lstruski/PMC/tree/master/demo/res) of PMC method. The following image presents dataset, which are consisted 3 (with one 2-dimensional and two 1-dimensional subspaces) subspaces.

<p align="center">
<img src="https://github.com/lstruski/PMC/blob/master/demo/orig_data.gif" width="500" height="500" />
</p>

#### How to run PMC on this data?
```
./build/PMC -f ../demo/data.txt -k 3 -i 20 -t 4 -l 40 -r 51 -o ../demo/res/
```
