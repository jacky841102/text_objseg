rm exp-referit/log/*global_context*
python exp-referit/exp_train_referit_seg_lowres.py 3 > exp-referit/log/train_referit_seg_lowres_global_context.log
python exp-referit/init_referit_seg_highres_from_lowres.py
python exp-referit/exp_train_referit_seg_highres.py 3 > exp-referit/log/train_referit_seg_highres_global_context.log
