REM python compare.py D:\data\folkdataset\4_mono folk D:\data\hooktheory_dataset\4_mono hook datasets_folk_vs_hook --mid_pattern=*.mid

REM python compare.py D:\thesis_code\model_folk100k_melody_bi2lstm32_attention\samples\lamb_transposed\ melody D:\thesis_code\model_folk100k_pianoroll_bi2lstm32_attention\samples\lamb_transposed pianoroll melody_vs_pianoroll

python compare.py D:\data\folkdataset\7_comparison training_set D:\thesis_code\model_folk100k_melody_bi2lstm64_attention\samples\blood_transposed model model_folk100k_melody_bi2lstm64_attention

python compare.py D:\data\hooktheory_dataset\7_comparison training_set D:\data\thesis_model2\model_hook100k_melody_bi2lstm64_attention\samples\blood_transposed\1.0 model hook100k_melody_bi2lstm64_attention_temp10

python compare.py D:\data\folkdataset\7_comparison training_set D:\thesis_code\model_folk100k_melody_bi2lstm32_attention\samples\blood_transposed\ model folk100k_melody_bi2lstm32_attention

python compare.py D:\data\folkdataset\7_comparison training_set D:\thesis_code\model_folk100k_pianoroll_bi2lstm32_attention\samples\blood_transposed\ model folk100k_pianoroll_bi2lstm32_attention

python compare.py D:\data\folkdataset\7_comparison training_set D:\thesis_code\model_folk100k_melody_bi2lstm64_attention\samples\blood_transposed\ model folk100k_melody_bi2lstm64_attention
