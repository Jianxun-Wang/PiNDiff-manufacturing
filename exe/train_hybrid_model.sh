python main.py  \
 --case_name    hybrid_LTP \
 --train_model  \
 --addNNDOC     \
 --NNDOC_layers 2 \
 --NNDOC_hsize  128 \
 --loss_flg     0,1,1e-8,1,1e-5 \
 --epochstart   0 \
 --nepoch       1000 \
#  --train_paranorm