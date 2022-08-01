from easydict import EasyDict as edict

config = edict()
config.loss = "arcface"
config.output = "arcface_ms1mv2_r50_ours"
config.network = "r50"
config.resume = False
config.pretrained_backbone_model = None
config.pretrained_partial_fc_path = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.1
config.verbose = 2000
config.dali = False
config.frequent = 10
config.score = None

config.rec = "/scratch1/feng/face_dataset/MS1M/v2_39m"
config.num_classes = 85742
config.num_image = 3870313
config.num_epoch = 20
config.warmup_epoch = 2
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]


config.rs_ratio = 0.75
config.pretrained_synthesis_model = "./pretrained_CFSM/WiderFace70K.pth"
config.epsilon = 0.314   
config.alpha = 0.314     
config.k = 1