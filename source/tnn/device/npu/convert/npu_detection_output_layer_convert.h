//// Tencent is pleased to support the open source community by making TNN available.
////
//// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
////
//// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//// in compliance with the License. You may obtain a copy of the License at
////
//// https://opensource.org/licenses/BSD-3-Clause
////
//// Unless required by applicable law or agreed to in writing, software distributed
//// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//// CONDITIONS OF ANY KIND, either express or implied. See the License for the
//// specific language governing permissions and limitations under the License.
//
//#ifndef TNN_SOURCE_TNN_DEVICE_NPU_CONVERT_NPU_DETECTION_OUTPUT_LAYER_CONVERT_H_
//#define TNN_SOURCE_TNN_DEVICE_NPU_CONVERT_NPU_DETECTION_OUTPUT_LAYER_CONVERT_H_
//
//#include <tnn/core/layer_type.h>
//#include <tnn/device/npu/convert/npu_base_layer_convert.h>
//#include <graph/op/all_ops.h>
//#include "graph/op/nn_defs.h"
//#include "npu_base_layer_convert.h"
//#include "npu_utils.h"
//namespace TNN_NS {
//class NpuDetectionOutputLayer : public NpuBaseLayer {
//public:
//    NpuDetectionOutputLayer(LayerType ignore) : NpuBaseLayer(LAYER_DETECTION_OUTPUT){};
//    virtual ~NpuDetectionOutputLayer(){};
//    Status SetInputShape(InputShapesMap input_shapes_map);
//
//protected:
//    virtual Status Convert();
//    // add for detection output
//    InputShapesMap input_shapes_map_;
//    std::vector<std::shared_ptr<ge::Operator>> weight_ops_;
//
//};
//
//}
//
//#endif  // TNN_SOURCE_TNN_DEVICE_NPU_CONVERT_NPU_DETECTION_OUTPUT_LAYER_CONVERT_H_
