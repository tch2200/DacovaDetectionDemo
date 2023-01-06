# Demo DacovaDetection for OpenVino

## Create and Export trained model to folder:
```
mkdir samples/weights/openvino
cd samples/weights/openvino
```
## Build docker
```
docker build openvino_cpp -t openvino_detection
```
## To compile in docker
```
docker run -it --rm -v $(pwd):/dacovadetection-openvino openvino_detection
cd openvino_cpp
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```
