# do training
bash tools/dist_trainval.sh  configs/vittad.py "5" 
#do testing
for i in {2..12..1}
do
   a="$i"00_weights
   echo $a
   CUDA_VISIBLE_DEVICES=5 python tools/thumos/test_af.py --framerate 8 configs/vittad.py workdir/vittad/epoch_$a.pth
done
