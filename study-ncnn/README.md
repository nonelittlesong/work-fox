## study-ncnn
[参考1](https://www.jianshu.com/p/f6b7bbb4fa96)  
[参考2](https://blog.csdn.net/yehui928186846/article/details/52787773)  

## 一、 安装protobuf
protobuf的[github](https://github.com/google/protobuf)  

安装以下工具：

  * autoconf
  * automake
  * libtool
  * make
  * g++
  * unzip
  
```
$ sudo apt-get install autoconf automake libtool curl make g++ unzip
```

安装protobuf：  
方法1：  
```
$ git clone https://github.com/google/protobuf.git
$ cd protobuf
$ git submodule update --init --recursive
$ ./autogen.sh

$ ./configure
$ make
$ make check
$ sudo make install
$ sudo ldconfig # refresh shared library cache.
```
方法2：
```
$ sudo pip install protobuf
```

## 二、 安装Android NDK

  1. [下载NDK](https://developer.android.google.cn/ndk/downloads/)
  2. 配置环境变量
  3. 配置as属性
  4. java测试代码
  5. 创建.h头文件
  6. 创建.c源文件
  7. 测试ndk环境

可以在Ansdoid Studio项目的local.propertiespei配置文件配置SDK和NDK的路径  
<br/>
配置系统环境变量：  
```
$ export NDK_HOME=/home/song/ndk_directory
$ export PATH=$NDK_HOME:$PATH
```
配置as（gradle.properties):
```
android.useDeprecatedNdk=true
```

## 三、 安装NCNN
下载ncnn  
```
$ git clone https://github.com/Tencent/ncnn --depth 1
$ cd ncnn
$ git fetch --unshallow
```
编译运行  
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ sudo make install
```
## 四、 caffe -> ncnn
[reference1](https://blog.csdn.net/Best_Coder/article/details/76201275)  
 
  * 搞定[caffe](https://github.com/nonelittlesong/study-caffe)
  * train.prototxt -> 旧deploy.prototxt
  * prototxt + caffemodel 旧模型转新模型
  * 根据情况修改new_deploy.prototxt
  * 新prototxt + caffemodel -> param + bin
  * 加密
  * 修改代码，读取模型
  * 编译
  

旧caffe -> 新caffe：  
```
$ ~/caffe/build/tools/upgrade_net_proto_text deploy.prototxt new_deploy.prototxt
$ ~/caffe/build/tools/upgrade_net_proto_binary bvlc_alexnet.caffemodel new_bvlc_alexnet.caffemodel
```
新caffe -> ncnn:  
```
$ ./caffe/caffe2ncnn new_deploy.prototxt new_bvlc_alexnet.caffemodel alexnet.param alexnet.bin
```
加密：  
```
$ ./ncnn2mem alexnet.param alexnet.bin alexnet.id.h alexnet.mem.h
```
生成alexnet.param.bin  
读取：  
```c++
// 对于加密文件的读取也和原来不同,在源码中,非加密param读取方式为
ncnn::Net net;
net.load_param("alexnet.param");
net.load_model("alexnet.bin");
// 加密param.bin读取方式为
ncnn::Net net;
net.load_param_bin("alexnet.param.bin");
net.load_model("alexnet.bin");
```
ncnn例程：  
```c++
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
#include "net.h"

static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;
    squeezenet.load_param("squeezenet_v1.1.param");
    squeezenet.load_model("squeezenet_v1.1.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Extractor ex = squeezenet.create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    cls_scores.resize(out.c);
    for (int j=0; j<out.c; j++)
    {
        const float* prob = out.data + out.cstep * j;
        cls_scores[j] = prob[0];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk, vector<int>& index_result, vector<float>& score_result)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        index_result.push_back(index);
        score_result.push_back(score);

        //fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

static int load_labels(string path, vector<string>& labels)
{
    FILE* fp = fopen(path.c_str(), "r");

    while (!feof(fp))
    {
        char str[1024];
        fgets(str, 1024, fp);  //¶ÁÈ¡Ò»ÐÐ
        string str_s(str);

        if (str_s.length() > 0)
        {
            for (int i = 0; i < str_s.length(); i++)
            {
                if (str_s[i] == ' ')
                {
                    string strr = str_s.substr(i, str_s.length() - i - 1);
                    labels.push_back(strr);
                    i = str_s.length();
                }
            }
        }
    }
    return 0;
}


int main(int argc, char** argv)
{
    const char* imagepath = argv[1];
    vector<string> labels;
    load_labels("synset_words.txt", labels);
    cv::Mat m = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<float> cls_scores;
    detect_squeezenet(m, cls_scores);

    vector<int> index;
    vector<float> score;
    print_topk(cls_scores, 3, index, score);


    for (int i = 0; i < index.size(); i++)
    {
       cv::putText(m, labels[index[i]], Point(50, 50 + 30 * i), CV_FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 100, 200), 2, 8);
    }

    imshow("m", m);
    imwrite("test_result.jpg", m);
    waitKey(0);

    return 0;
}
```
## 五、 ncnn -> android
[reference1](https://blog.csdn.net/qq_36982160/article/details/79931741)  
https://github.com/Tencent/ncnn/wiki/cmake-%E6%89%93%E5%8C%85-android-sdk  
**编译ncnn sdk：**  
我们需要将ncnn打包，这样才能在android ndk的代码中使用include：  
```
$ mkdir build-android
$ cd build-android
$ cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON \
      _DANDROID_PLATFORM=android-14 ..
$ make
$ make install
$ make package
```
ANDROID_ABI是架构的名字， "armeabi-v7a"支持绝大部分手机硬件  
ANDROID_ARM_NEON是否使用NEON指令集， 设为ON支持绝大部分手机硬件  
ANDROID_PALTFORM指定最低系统v版本， "android-14"就是android 4.0  
## 六、 Troubleshooting
#### ncnn在AS上build失败
构建项目时应该使用自己编译的ncnn，否则会因为路径问题出错。  
