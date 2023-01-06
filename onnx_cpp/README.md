# Demo DacovaDetection for ONNX

## Create and Export trained model to folder:
```
mkdir samples/weights/onnx
cd samples/weights/onnx
```
## Build docker
```
docker build onnx_cpp -t onnx_detection
```

## To compile in docker
```
docker run -it --rm -v $(pwd):/mnt onnx_detection
cd /mnt/onnx_cpp 
mkdir build
cd build 
cmake ../ -O ./
make
./main