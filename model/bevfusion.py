from .fusion_template import FusionTemplate
import Swintransformer, generalized_lss, depth_lss

class BevFusion(FusionTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 
            'image_backbone','neck','vtransform',
            'fuser','backbone_2d','dense_head'
        ]
        self.module_list = self.build_networks()
    def build_image_backbone(self, model_info_dict):

        image_backbone_module =  Swintransformer.SwinTransformer(
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict
    
    def build_neck(self,model_info_dict):
        neck_module = generalized_lss.GeneralizedLSSFPN(
            model_cfg=self.model_cfg.NECK
        )
        model_info_dict['module_list'].append(neck_module)

        return neck_module, model_info_dict
    
    def build_vtransform(self,model_info_dict):

        vtransform_module = depth_lss.DepthLSSTransform(
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)
    
        return vtransform_module, model_info_dict

    def build_fuser(self, model_info_dict):

        fuser_module = convfuser.ConvFuser(
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.FUSER.OUT_CHANNEL
        return fuser_module, model_info_dict

    def forward(self, batch_dict):

        for i,cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
