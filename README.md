# Demo DacovaDetection

## Download original weight to folder:
```
mkdir samples/weights/tf
cd samples/weights/tf
```
## Build docker
```
docker build cpp -t openvino_detection
docker run -it --rm -v $(pwd):/dacovadetection-openvino openvino_detection
```
## To compile in docker
```
cd cpp
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```
## To Run with python
```
pip install -r requirements.txt
python openvino_demo.py
python openvino_video.py
```

