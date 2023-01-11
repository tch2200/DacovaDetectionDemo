# Demo DacovaDetection

## Create and Export trained model to folder:
```
mkdir samples/weights/openvino
mkdir samples/weights/onnx
```
## To Run with openvino python
```
pip install -r requirements.txt
python openvino_demo.py
python openvino_video.py
```
## To Run with onnx python
```
python onnx_demo.py
```
## Build openvino docker
```
docker build openvino_cpp -t openvino_detection
docker run -it --rm -v $(pwd):/openvino_detection openvino_detection
```
## To compile in docker
```
cd openvino_cpp
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```
## Build onnx docker
```
docker build onnx_cpp -t onnx_detection
docker run -it --rm -v $(pwd):/onnx_detection onnx_detection
```
## To compile in docker
```
cd onnx_cpp
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```

