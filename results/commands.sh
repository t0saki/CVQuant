python3 main.py --model resnet18_low_rank --dataset cifar10 --methods static qat --enable-distillation --teacher-model resnet50 --batch-size 32 --calibration-size 1000 --evaluation-size 500 --device cpu --output-dir ./demo_results/quick_distillation --backend qnnpack


python main.py --device cpu --backend qnnpack --methods static qat --model resnet18_low_rank --calibration-size 1000 --evaluation-size 5000 --enable-distillation --teacher-model efficientnetv2_s > distill_resnet18_lrf.log 2>&1 && python main.py --device cpu --backend qnnpack --methods static qat --model resnet50_quantizable --calibration-size 1000 --evaluation-size 5000 --enable-distillation --teacher-model efficientnetv2_m > distill_resnet50.log 2>&1


python main.py --device mps --backend qnnpack --methods static qat --model resnet50 --dataset imagenet --calibration-size 50000 --evaluation-size 50000 --batch-size 8 --enable-finetuning --enable-distillation --teacher-model efficientnetv2_m > resnet_50_imagenet_mps.log

cd data
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
cd ../
python main.py --device mps --backend qnnpack --methods static qat --model resnet50 --dataset imagenet --calibration-size 50000 --evaluation-size 50000 --batch-size 8 --enable-finetuning --enable-distillation --teacher-model efficientnetv2_m > resnet_50_imagenet_mps.log

python main.py --device mps --backend qnnpack --methods static qat --model resnet18_quantizable --dataset imagenet --calibration-size 50000 --evaluation-size 50000 --batch-size 8 --enable-finetuning --enable-distillation --teacher-model resnet50_quantizable > resnet_18_imagenet_mps.log