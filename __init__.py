from ComfyUI_hunyuanloom_nodes.nodes.modify_hy_model_node import ConfigureModifiedHYNode
from ComfyUI_hunyuanloom_nodes.nodes.hy_model_pred_nodes import HYInverseModelSamplingPredNode, HYReverseModelSamplingPredNode
from ComfyUI_hunyuanloom_nodes.nodes.rectified_sampler_nodes import HYForwardODESamplerNode, HYReverseODESamplerNode
from ComfyUI_hunyuanloom_nodes.nodes.flowedit_nodes import HYFlowEditGuiderNode, HYFlowEditGuiderAdvNode, HYFlowEditSamplerNode, HYFlowEditGuiderCFGNode, HYFlowEditGuiderCFGAdvNode

from ComfyUI_hunyuanloom_nodes.nodes.hy_regional_cond_nodes import HYApplyRegionalCondsNode, HYCreateRegionalCondNode
from ComfyUI_hunyuanloom_nodes.nodes.hy_attn_override_node import HYAttnOverrideNode

from ComfyUI_hunyuanloom_nodes.nodes.hy_feta_enhance_node import HYFetaEnhanceNode

from ComfyUI_hunyuanloom_nodes.nodes.wrapper_flow_edit_nodes import HyVideoFlowEditSamplerNode


NODE_CLASS_MAPPINGS = {
    "ConfigureModifiedHY": ConfigureModifiedHYNode,
    # RF-Inversion
    "HYInverseModelSamplingPred": HYInverseModelSamplingPredNode,
    "HYReverseModelSamplingPred": HYReverseModelSamplingPredNode,
    "HYForwardODESampler": HYForwardODESamplerNode,
    "HYReverseODESampler": HYReverseODESamplerNode,
    # FlowEdit
    "HYFlowEditGuider": HYFlowEditGuiderNode,
    "HYFlowEditGuiderAdv": HYFlowEditGuiderAdvNode,
    "HYFlowEditGuiderCFG": HYFlowEditGuiderCFGNode,
    "HYFlowEditGuiderCFGAdv": HYFlowEditGuiderCFGAdvNode,
    "HYFlowEditSampler": HYFlowEditSamplerNode,
    # Regional
    "HYApplyRegionalConds": HYApplyRegionalCondsNode,
    "HYCreateRegionalCond": HYCreateRegionalCondNode,
    "HYAttnOverride": HYAttnOverrideNode,
    # Enhance
    "HYFetaEnhance": HYFetaEnhanceNode,
    # Wrapper
    "HyVideoFlowEditSamplerWrapper": HyVideoFlowEditSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConfigureModifiedHY": "Modify Hunyuan Model",
    # RF-Inversion
    "HYInverseModelSamplingPred": "HY Inverse Model Pred",
    "HYReverseModelSamplingPred": "HY Reverse Model Pred",
    "HYForwardODESampler": "HY RF-Inv Forward Sampler",
    "HYReverseODESampler": "HY RF-Inv Reverse Sampler",
    # FlowEdit
    "HYFlowEditGuider": "HY FlowEdit Guider",
    "HYFlowEditGuiderCFG": "HY FlowEdit Guider CFG",
    "HYFlowEditGuiderCFGAdv": "HY FlowEdit Guider CFG Adv.",
    "HYFlowEditSampler": "HY FlowEdit Sampler",
    # Regional
    "HYApplyRegionalConds": "HY Apply Regional Conds",
    "HYCreateRegionalCond": "HY Create Regional Cond",
    "HYAttnOverride": "HY Attention Override",
    # Enhance
    "HYFetaEnhance": "HY Feta Enhance",
    # Wrapper
    "HyVideoFlowEditSamplerWrapper": "HunyuanVideo Flow Edit Sampler (Wrapper)",
}
