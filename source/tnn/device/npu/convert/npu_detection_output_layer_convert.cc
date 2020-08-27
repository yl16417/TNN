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
//#include "npu_detection_output_layer_convert.h"
//#include <graph/op/all_ops.h>
//#include "graph/op/nn_defs.h"
//#include "npu_utils.h"
//
//namespace TNN_NS {
//
//Status NpuDetectionOutputLayer::SetInputShape(InputShapesMap input_shapes_map) {
//    input_shapes_map_ = input_shapes_map;
//    return TNN_OK;
//}
//
//Status NpuDetectionOutputLayer::Convert() {
//    // parameter and weight of the DetectionOutput layer
//    auto param = dynamic_cast<DetectionOutputLayerParam *>(param_);
//    CHECK_PARAM_NULL(param);
//
//    std::vector<int> im_info;
//    for (auto it = input_shapes_map_.begin(); it != input_shapes_map_.end(); it++) {
//        if (it->second[2] > 0 && it->second[3] > 0) {
//            int h = it->second[2];
//            int w = it->second[3];
//            printf("the height wide %d %d\n,", h, w);
//            im_info.push_back(h);
//            im_info.push_back(w);
//        }
//    }
//
//    if (im_info.size() != 2)
//        return Status(TNNERR_MODEL_ERR,
//                      "EMPTY I"
//                      "NPUT SHAPE");
//
//    auto output = std::make_shared<ge::op::FSRDetectionOutput>(outputs_name_[0]);
//    output->set_input_score(*input_ops_[1]->GetOperator());
//    output->set_input_bbox_pred(*input_ops_[2]->GetOperator());
//    output->set_input_rois(*input_ops_[0]->GetOperator());
//    output->set_attr_img_h(im_info[0]);
//    output->set_attr_img_w(im_info[1]);
//    output->set_attr_num_classes(param->num_classes);
//    output->set_attr_confidence_threshold(param->confidence_threshold);
//    output->set_attr_nms_threshold(param->nms_param.nms_threshold);
//
//    ADD_OUTPUT_OP(output)
//}
//REGISTER_NPU_LAYER(DetectionOutput, LAYER_DETECTION_OUTPUT)
//}  // namespace TNN_NS