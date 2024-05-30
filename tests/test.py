'''
为流通而造的船，遇到港口也会停泊，所以璃月是一切财富沉淀的地方。.wav

python api_v2.py -a 0.0.0.0 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml

(bert-vits2) root@diy01:/home/diy/congcan/source/GPT-SoVITS# curl -XPOST -d @data.json http://127.0.0.1:9880/tts -H "Content-Type: application/json"
Warning: Binary output can mess up your terminal. Use "--output -" to tell 
Warning: curl to output it to your terminal anyway, or consider "--output 
Warning: <FILE>" to save to a file.


http://127.0.0.1:9880/tts?text=%E5%85%88%E5%B8%9D%E5%88%9B%E4%B8%9A%E6%9C%AA%E5%8D%8A%E8%80%8C%E4%B8%AD%E9%81%93%E5%B4%A9%E6%AE%82%EF%BC%8C%E4%BB%8A%E5%A4%A9%E4%B8%8B%E4%B8%89%E5%88%86%EF%BC%8C%E7%9B%8A%E5%B7%9E%E7%96%B2%E5%BC%8A%EF%BC%8C%E6%AD%A4%E8%AF%9A%E5%8D%B1%E6%80%A5%E5%AD%98%E4%BA%A1%E4%B9%8B%E7%A7%8B%E4%B9%9F%E3%80%82&text_lang=zh&ref_audio_path=tests/zl.000.wav&prompt_lang=zh&prompt_text=%E4%B8%BA%E6%B5%81%E9%80%9A%E8%80%8C%E9%80%A0%E7%9A%84%E8%88%B9%EF%BC%8C%E9%81%87%E5%88%B0%E6%B8%AF%E5%8F%A3%E4%B9%9F%E4%BC%9A%E5%81%9C%E6%B3%8A%EF%BC%8C%E6%89%80%E4%BB%A5%E7%92%83%E6%9C%88%E6%98%AF%E4%B8%80%E5%88%87%E8%B4%A2%E5%AF%8C%E6%B2%89%E6%B7%80%E7%9A%84%E5%9C%B0%E6%96%B9%E3%80%82&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true


http://127.0.0.1:9880/tts?text=《杀死一只知更鸟》是美国女作家哈珀·李发表于1960年的长篇小说。该小说讲述一个名叫汤姆·鲁滨逊的年轻人,被人诬告犯了强奸罪后,只是因为是一个黑人,辩护律师阿蒂克斯·芬奇尽管握有汤姆不是强奸犯的证据,都无法阻止陪审团给出汤姆有罪的结论。此一妄加之罪,导致汤姆死于乱枪之下。虽然故事题材涉及种族不平等与强暴等严肃议题,其文风仍温暖风趣。&text_lang=zh&ref_audio_path=tests/zl.000.wav&prompt_lang=zh&prompt_text=%E4%B8%BA%E6%B5%81%E9%80%9A%E8%80%8C%E9%80%A0%E7%9A%84%E8%88%B9%EF%BC%8C%E9%81%87%E5%88%B0%E6%B8%AF%E5%8F%A3%E4%B9%9F%E4%BC%9A%E5%81%9C%E6%B3%8A%EF%BC%8C%E6%89%80%E4%BB%A5%E7%92%83%E6%9C%88%E6%98%AF%E4%B8%80%E5%88%87%E8%B4%A2%E5%AF%8C%E6%B2%89%E6%B7%80%E7%9A%84%E5%9C%B0%E6%96%B9%E3%80%82&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true



'''



'''

docker run --rm -it --gpus=all --env=is_half=False \
	--volume=./GPT-SoVITS-DockerTest/logs:/workspace/logs \
	--volume=./GPT-SoVITS-DockerTest/output:/workspace/output \
	--volume=./GPT-SoVITS-DockerTest/SoVITS_weights:/workspace/SoVITS_weights \
	--workdir=/workspace \
	-p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 \
	--shm-size="16G" -d breakstring/gpt-sovits


'''



'''
# HEX
# 编码  
echo -n "为流通而造的船，遇到港口也会停泊，所以璃月是一切财富沉淀的地方。" | xxd -p -u  
  
# 解码  
echo -n "E4B8BAE6B581E9809AE8808CE980A0E79A84E888B9EFBC8CE98187E588B0E6B8AFE58FA3E4B99FE4BC9AE5819CE6B38AEFBC8CE68980E4BBA5E79283E69C88E698AFE4B880E58887E8B4A2E5AF8CE6B289E6B780E79A84E59CB0E696B9E38082" | xxd -r -p

'''



