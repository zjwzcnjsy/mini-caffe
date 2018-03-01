#include <string>

#include <caffe/net.hpp>
#include <caffe/profiler.hpp>

int main(int argc, char *argv[]) {
  CHECK_GE(argc, 4) << "[Usage]: ./benchmark net.prototxt [net.caffemodel] iterations gpu_id";
  CHECK_LE(argc, 5) << "[Usage]: ./benchmark net.prototxt [net.caffemodel] iterations gpu_id";
  std::string proto = argv[1];
  std::string model = "";
  int iters = 1;
  int gpu_id = -1;
  if (argc == 4) {
	  iters = std::stoi(argv[2]);
	  gpu_id = std::stoi(argv[3]);
  }
  else if (argc == 5) {
	  model = argv[2];
	  iters = std::stoi(argv[3]);
	  gpu_id = std::stoi(argv[4]);
  }
  LOG(INFO) << "net prototxt: " << proto;
  if (argc == 5) {
	  LOG(INFO) << "net caffemodel: " << model;
  }
  LOG(INFO) << "net forward iterations: " << iters;
  LOG(INFO) << "run on device " << gpu_id;

  if (gpu_id >= 0 && caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, gpu_id);
  }
  else {
    gpu_id = -1;
  }

  caffe::Net net(proto);
  if (argc == 5) {
	  net.CopyTrainedLayersFrom(model);
  }
  caffe::Profiler* profiler = caffe::Profiler::Get();
  profiler->TurnON();
  for (int i = 0; i < iters; i++) {
    net.Forward();
  }
  profiler->TurnOFF();
  profiler->DumpProfile("./profile.json");
  return 0;
}
