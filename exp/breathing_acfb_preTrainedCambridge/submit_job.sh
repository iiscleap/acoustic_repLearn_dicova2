
tag="PRETRAIN_FB_CAMBRIDGE"
gpu_id=1
mkdir -p $tag
cp run_cls.sh $tag/

./run_cls.sh $tag

